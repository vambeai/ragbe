"""
Document service — orchestrates PDF upload, conversion, and deletion.

# VERSION 3 — Unified ProcessPoolExecutor for all CPU-bound converters
#
# All CPU-bound converters (PyMuPDF, Docling, MarkItDown) now run in a shared
# ProcessPoolExecutor via convert_in_process(). VLM and Cloud remain thread-based (I/O).
# Worker processes are initialised once via _init_cpu_worker() which pre-loads
# the DocumentService and Docling ML models, avoiding per-job reload cost.
# Top-level functions are required because child processes serialise them via
# pickle, which does not support local/nested functions.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Type

from backend.config import get_settings

logger = logging.getLogger(__name__)

from fastapi import HTTPException, UploadFile

from backend.converters.vlm_checkpoint import CheckpointStore, discard_all_for_stem
from backend.services.enrichment_checkpoint import discard_all_for_stem as discard_enrich_for_stem
from backend.services.document_summary import discard_all_for_stem as discard_summary_for_stem
from backend.converters.base import PDFConverter
from backend.converters.cloud import CloudConverter
from backend.converters.docling import DoclingConverter
from backend.converters.liteparse import LiteParseConverter
from backend.converters.markitdown import MarkItDownConverter
from backend.converters.pymupdf import PyMuPDFConverter
from backend.converters.vlm import VLMConverter
from backend.models.schemas import (
    CloudSettings,
    ConvertResponse,
    ConverterType,
    DeleteResponse,
    DocumentInfo,
    MarkdownContentResponse,
    MarkdownVersion,
    MdToPdfResponse,
    MultiUploadResponse,
    UploadFileResult,
    VLMSettings,
)
from backend.utils.naming import KNOWN_CONVERTERS as _KNOWN_CONVERTERS, normalise_converter as _normalise_converter
from backend.utils.path import safe_child_path, safe_filename, safe_stem

_ALLOWED_EXTENSIONS = {".pdf", ".md"}

# Detection pattern for VLM partial-run placeholders.  Kept module-level so
# the compiled regex is reused across many calls to ``list_markdown_versions``
# in a single batch.
import re as _re
_FAILURE_MARKER_RE = _re.compile(r"<!--\s*page\s+\d+\s+failed:")


def _file_has_failure_markers(path: Path) -> bool:
    """Return True iff *path* contains at least one VLM failure placeholder.

    Reads the file lazily until the first hit so very large documents don't
    pay for a full read just to flag them.  Returns False on any read error
    (missing file, encoding glitch) — the absence of a positive signal is
    safer than surfacing a phantom warning.
    """
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if _FAILURE_MARKER_RE.search(line):
                    return True
        return False
    except OSError:
        return False


def _md_files_for_stem(mds_dir: Path, stem: str) -> list[Path]:
    """Return every Markdown file on disk that belongs to *stem*.

    Includes both converted variants (``{stem}_{converter}.md``) and the
    legacy / uploaded variant whose filename equals ``{stem}.md``.  The
    caller is responsible for distinguishing the two via filename inspection.
    """
    if not mds_dir.exists():
        return []
    matches: list[Path] = []
    for f in mds_dir.glob(f"{stem}*.md"):
        if f.stem == stem:
            matches.append(f)
            continue
        suffix = f.stem[len(stem):]
        if not suffix.startswith("_"):
            continue
        token = suffix[1:]
        # Only count it as belonging to *stem* if the suffix matches a known
        # converter — otherwise the file just happens to share the prefix.
        if token in _KNOWN_CONVERTERS:
            matches.append(f)
    return matches


def find_markdown_for_document(
    filename: str,
    mds_dir: Path,
    md_filename: str | None = None,
) -> Path | None:
    """Locate the Markdown file the chunking worker should read.

    When *md_filename* is provided and matches the document, return that
    exact file — this lets the user pick a specific MD variant from the
    frontend dropdown and chunk against it instead of "whichever happens
    to be first on disk".  Otherwise fall back to the first available
    converted variant, then the legacy/uploaded ``{stem}.md``.
    """
    stem = Path(filename).stem

    # Honour the explicit selection when it points at a real file under
    # mds_dir AND belongs to the same document stem (defends against
    # path-traversal and cross-document leakage).
    if md_filename:
        candidate = mds_dir / md_filename
        if candidate.exists() and candidate.parent == mds_dir and candidate.stem.startswith(stem):
            return candidate

    # If the input itself is an .md filename, prefer that exact file.
    direct = mds_dir / filename if filename.lower().endswith(".md") else None
    if direct is not None and direct.exists():
        return direct

    candidates = _md_files_for_stem(mds_dir, stem)
    if not candidates:
        return None
    # Prefer converted variants over the legacy/uploaded one for
    # deterministic ordering.
    converted = [p for p in candidates if p.stem != stem]
    if converted:
        return sorted(converted)[0]
    return candidates[0]


# ---------------------------------------------------------------------------
# Top-level worker functions for ProcessPoolExecutor
# ---------------------------------------------------------------------------

_worker_svc: "DocumentService | None" = None


def _init_cpu_worker() -> None:
    """Initializer executed once per worker process at startup."""
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(processName)s] %(levelname)s %(name)s — %(message)s",
    )

    global _worker_svc
    _worker_svc = DocumentService()

    try:
        from docling.document_converter import DocumentConverter
        DocumentConverter()
    except Exception as exc:
        _logging.getLogger(__name__).warning("Docling pre-load failed in worker: %s", exc)


def convert_in_process(filename: str, converter_type: ConverterType) -> ConvertResponse:
    """Run a CPU-bound PDF→Markdown conversion in a worker process."""
    if _worker_svc is None:
        raise RuntimeError("Worker process not initialised — _init_cpu_worker did not complete")
    return _worker_svc.convert_to_markdown(filename, converter_type=converter_type)


def convert_md_to_pdf_in_process(md_filename: str) -> MdToPdfResponse:
    """Run a Markdown→PDF conversion in a worker process."""
    if _worker_svc is None:
        raise RuntimeError("Worker process not initialised — _init_cpu_worker did not complete")
    return _worker_svc.convert_md_to_pdf(md_filename)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stem(filename: str) -> str:
    return Path(filename).stem


def _dest_dir(filename: str, pdfs_dir: Path, mds_dir: Path) -> Path:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return pdfs_dir
    if ext == ".md":
        return mds_dir
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type '{ext}'. Allowed: .pdf, .md",
    )


def _build_converter(
    converter_type: ConverterType,
    vlm_settings: VLMSettings | None,
    cloud_settings: CloudSettings | None,
    on_progress: Callable[[int, int], None] | None = None,
    stop_event: threading.Event | None = None,
    checkpoint_store: CheckpointStore | None = None,
) -> PDFConverter:
    """Instantiate the requested converter, forwarding runtime settings when relevant."""
    if converter_type == ConverterType.vlm:
        kwargs: dict = vlm_settings.model_dump(exclude_none=True) if vlm_settings else {}
        # ``use_checkpoint`` is a service-level flag (controls whether the
        # checkpoint store is constructed/cleared) and is not part of
        # VLMConverter's constructor signature.  Drop it before unpacking so
        # the converter doesn't reject the kwarg.
        kwargs.pop("use_checkpoint", None)
        if on_progress:
            kwargs["on_progress"] = on_progress
        if stop_event is not None:
            kwargs["stop_event"] = stop_event
        if checkpoint_store is not None:
            kwargs["checkpoint_store"] = checkpoint_store
        return VLMConverter(**kwargs)

    if converter_type == ConverterType.cloud:
        kwargs = cloud_settings.model_dump(exclude_none=True) if cloud_settings else {}
        if on_progress:
            kwargs["on_progress"] = on_progress
        if stop_event is not None:
            kwargs["stop_event"] = stop_event
        return CloudConverter(**kwargs)

    return _CONVERTER_MAP[converter_type]()


# Converter registry — maps enum value to class (excludes VLM and Cloud which need runtime args)
_CONVERTER_MAP: dict[ConverterType, Type[PDFConverter]] = {
    ConverterType.pymupdf: PyMuPDFConverter,
    ConverterType.docling: DoclingConverter,
    ConverterType.markitdown: MarkItDownConverter,
    ConverterType.liteparse: LiteParseConverter,
}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DocumentService:
    """Handles all document-level operations: listing, uploading, converting, deleting."""

    def __init__(self) -> None:
        s = get_settings()
        self._pdfs_dir = Path(s.PDFS_DIR)
        self._mds_dir = Path(s.MDS_DIR)
        self._chunks_dir = Path(s.CHUNKS_DIR)

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------

    def _discard_derived_caches(self, stem: str) -> None:
        """Wipe every on-disk cache derived from a source document.

        Centralises the three cascade calls (VLM checkpoints, enrichment
        checkpoints, document summaries) so adding a new cache tier later
        only has to be wired into one place — call sites (upload-overwrite,
        delete) can't accidentally forget one of the three.  Each underlying
        ``discard_all_for_stem`` is independently best-effort: I/O failures
        are logged but never raised, so cleanup of one tier never blocks
        the others.
        """
        discard_all_for_stem(stem, self._mds_dir)
        discard_enrich_for_stem(stem, self._mds_dir)
        discard_summary_for_stem(stem, self._mds_dir)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _is_md_associated_with_pdf(self, md_file: Path, pdf_stems: set[str]) -> bool:
        """True if *md_file* belongs to one of the PDFs identified by *pdf_stems*.

        A file belongs to a PDF when its stem either equals the PDF stem
        exactly (legacy / matching upload) or starts with ``{pdf_stem}_`` and
        the trailing token is a known converter name.
        """
        for pdf_stem in pdf_stems:
            if md_file.stem == pdf_stem:
                return True
            prefix = f"{pdf_stem}_"
            if md_file.stem.startswith(prefix):
                token = md_file.stem[len(prefix):]
                if token in _KNOWN_CONVERTERS:
                    return True
        return False

    def list_documents(self) -> list[str]:
        results: list[str] = []
        pdf_stems: set[str] = set()

        if self._pdfs_dir.exists():
            for f in self._pdfs_dir.glob("*.pdf"):
                results.append(f.name)
                pdf_stems.add(f.stem)

        if self._mds_dir.exists():
            for f in self._mds_dir.glob("*.md"):
                if not self._is_md_associated_with_pdf(f, pdf_stems):
                    results.append(f.name)

        return sorted(results)

    def list_documents_metadata(self) -> list[dict]:
        """Return per-document metadata for the sidebar listing.

        ``has_failures`` is true when EITHER (a) a VLM checkpoint directory
        survives on disk for the document — the usual "partial conversion"
        signal — OR (b) the VLM markdown variant on disk contains one or
        more failure placeholders.  Case (b) covers two paths that case
        (a) misses: a conversion where *every* page failed (no successes,
        therefore the checkpoint dir was never created) and a run whose
        checkpoint was manually deleted while the placeholder-laden MD
        remained.  Keeping both signals in sync keeps the sidebar warning
        aligned with what the dropdown and in-viewer banner show.
        Other converters never emit placeholders, so this check is
        VLM-scoped.
        """
        results = []
        pdf_stems: set[str] = set()

        if self._pdfs_dir.exists():
            for f in sorted(self._pdfs_dir.glob("*.pdf")):
                stem = f.stem
                has_md = bool(_md_files_for_stem(self._mds_dir, stem))
                has_failures = CheckpointStore(stem, self._mds_dir).exists()
                if not has_failures:
                    vlm_md = self._mds_dir / f"{stem}_vlm.md"
                    if vlm_md.exists() and _file_has_failure_markers(vlm_md):
                        has_failures = True
                results.append({
                    "filename": f.name,
                    "has_markdown": has_md,
                    "has_failures": has_failures,
                })
                pdf_stems.add(stem)

        if self._mds_dir.exists():
            for f in sorted(self._mds_dir.glob("*.md")):
                if not self._is_md_associated_with_pdf(f, pdf_stems):
                    # Standalone .md uploads can't have a VLM checkpoint —
                    # they were never produced by a per-page converter.
                    results.append({
                        "filename": f.name,
                        "has_markdown": True,
                        "has_failures": False,
                    })

        return results

    def get_document(self, filename: str) -> DocumentInfo:
        filename = safe_filename(filename, "document name")
        ext = Path(filename).suffix.lower()

        if ext == ".md":
            md_path = self._mds_dir / filename
            if not md_path.exists():
                raise HTTPException(status_code=404, detail=f"MD '{filename}' not found")
            md_content = md_path.read_text(encoding="utf-8")
            pdf_filename = f"{_stem(filename)}.pdf"
            has_pdf = (self._pdfs_dir / pdf_filename).exists()
            return DocumentInfo(
                pdf_filename=pdf_filename,
                md_filename=filename,
                md_content=md_content,
                has_markdown=True,
                has_pdf=has_pdf,
            )

        pdf_path = self._pdfs_dir / filename
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF '{filename}' not found")

        # Pick the first available Markdown variant for this PDF as the
        # "default" version returned in DocumentInfo.  The caller can switch
        # to other versions using GET /documents/{name}/markdowns.
        stem = _stem(filename)
        candidates = _md_files_for_stem(self._mds_dir, stem)
        if candidates:
            converted = sorted([p for p in candidates if p.stem != stem])
            chosen = converted[0] if converted else candidates[0]
            md_filename = chosen.name
            md_content = chosen.read_text(encoding="utf-8")
            has_markdown = True
        else:
            # No file yet — surface a placeholder filename so existing
            # frontend logic (which uses md_filename for save targets) keeps
            # working until the user converts.
            md_filename = f"{stem}.md"
            md_content = ""
            has_markdown = False

        return DocumentInfo(
            pdf_filename=filename,
            md_filename=md_filename,
            md_content=md_content,
            has_markdown=has_markdown,
            has_pdf=True,
        )

    def list_markdown_versions(self, document_name: str) -> list[MarkdownVersion]:
        """Return every Markdown version on disk for *document_name*.

        ``document_name`` may be a PDF filename, an MD filename, or a bare
        stem.  Converted versions are reported with ``source="converted"``
        and the parsed converter token; uploaded files (whose name does not
        match the ``{stem}_{converter}.md`` pattern) are reported with
        ``source="uploaded"`` and ``converter=None``.

        Each version also carries ``has_failures``: true iff the file
        contains one or more ``<!-- page N failed: … -->`` placeholders
        emitted by the VLM converter after a partial run.  The frontend
        version picker uses this flag to mark the affected variant so the
        user can tell at a glance which one is incomplete.
        """
        stem = safe_stem(document_name)
        if not self._mds_dir.exists():
            return []

        versions: list[MarkdownVersion] = []
        # Converted variants
        for converter in _KNOWN_CONVERTERS:
            candidate = self._mds_dir / f"{stem}_{converter}.md"
            if candidate.exists():
                versions.append(MarkdownVersion(
                    filename=candidate.name,
                    source="converted",
                    converter=converter,
                    file_path=str(candidate),
                    has_failures=_file_has_failure_markers(candidate),
                ))

        # Uploaded / legacy: any .md whose name does not match the converter
        # suffix pattern but starts with the stem.  This covers both the
        # exact-match case ({stem}.md) and idiosyncratic uploaded filenames
        # that share the stem.
        for f in sorted(self._mds_dir.glob(f"{stem}*.md")):
            if f.stem == stem:
                versions.append(MarkdownVersion(
                    filename=f.name,
                    source="uploaded",
                    converter=None,
                    file_path=str(f),
                    has_failures=_file_has_failure_markers(f),
                ))
                continue
            suffix = f.stem[len(stem):]
            if suffix.startswith("_") and suffix[1:] in _KNOWN_CONVERTERS:
                continue
            # Anything else with the stem prefix that wasn't picked up above
            # is treated as uploaded so the user still sees it.
            if f.stem.startswith(stem):
                versions.append(MarkdownVersion(
                    filename=f.name,
                    source="uploaded",
                    converter=None,
                    file_path=str(f),
                    has_failures=_file_has_failure_markers(f),
                ))

        return versions

    def get_markdown_content(self, document_name: str, identifier: str) -> MarkdownContentResponse:
        """Return the content of one specific Markdown version.

        ``identifier`` may be either a converter name (resolves to
        ``{stem}_{identifier}.md``) or the original on-disk filename for
        uploaded MDs.  This dual lookup keeps the URL natural for both
        cases without exposing internal naming details to the client.
        """
        stem = safe_stem(document_name)

        # 1) Converter-name lookup
        if identifier in _KNOWN_CONVERTERS:
            candidate = safe_child_path(self._mds_dir, f"{stem}_{identifier}.md", description="identifier")
            if candidate.exists():
                return MarkdownContentResponse(
                    filename=candidate.name,
                    source="converted",
                    converter=identifier,
                    content=candidate.read_text(encoding="utf-8"),
                )

        # 2) Direct filename lookup
        if identifier.endswith(".md"):
            candidate = safe_child_path(self._mds_dir, identifier, description="identifier")
            if candidate.exists() and candidate.stem.startswith(stem):
                # Determine source from the filename pattern
                suffix = candidate.stem[len(stem):]
                if suffix.startswith("_") and suffix[1:] in _KNOWN_CONVERTERS:
                    return MarkdownContentResponse(
                        filename=candidate.name,
                        source="converted",
                        converter=suffix[1:],
                        content=candidate.read_text(encoding="utf-8"),
                    )
                return MarkdownContentResponse(
                    filename=candidate.name,
                    source="uploaded",
                    converter=None,
                    content=candidate.read_text(encoding="utf-8"),
                )

        raise HTTPException(
            status_code=404,
            detail=f"Markdown version '{identifier}' not found for '{document_name}'",
        )

    def get_vlm_checkpoint_info(self, document_name: str) -> dict:
        """Return the VLM checkpoint state for ``document_name``.

        Used by the frontend to surface a "Resume available" indicator.  The
        document need not currently exist on disk — if the PDF was deleted
        with orphan checkpoint files left over (should not happen with the
        regular ``delete_document`` flow, but kept defensive), ``exists`` is
        reported truthfully and the caller can decide whether to discard.
        """
        stem = safe_stem(document_name)
        store = CheckpointStore(stem, self._mds_dir)
        # 1-indexed for the UI (page numbers shown to users start at 1).
        completed = [p + 1 for p in store.completed_pages()]
        return {
            "document_name": document_name,
            "converter": "vlm",
            "exists": bool(completed),
            "completed_pages": completed,
        }

    def get_pdf_path(self, filename: str) -> Path:
        filename = safe_filename(filename, "PDF filename")
        pdf_path = self._pdfs_dir / filename
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF '{filename}' not found")
        return pdf_path

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_file(self, file: UploadFile) -> None:
        import filetype

        name = safe_filename(file.filename or "", "upload filename")
        dest_dir = _dest_dir(name, self._pdfs_dir, self._mds_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        settings = get_settings()
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        read_chunk_bytes = settings.UPLOAD_READ_CHUNK_BYTES
        magic_probe_bytes = settings.PDF_MAGIC_PROBE_BYTES

        dest_path = dest_dir / name
        size = 0
        buf_for_magic = bytearray()
        is_pdf_ext = name.lower().endswith(".pdf")
        # Detect "uploading over an existing PDF" up-front; the actual stale-
        # derivative cleanup is deferred to after the new file is fully
        # written and validated so a failed upload doesn't destroy
        # converted artifacts that still match the previous on-disk PDF.
        was_overwrite = is_pdf_ext and dest_path.exists()

        try:
            with open(dest_path, "wb") as out:
                while True:
                    chunk = file.file.read(read_chunk_bytes)
                    if not chunk:
                        break
                    size += len(chunk)
                    if max_bytes > 0 and size > max_bytes:
                        dest_path.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                f"'{name}' exceeds the {settings.MAX_FILE_SIZE_MB} MB upload limit "
                                f"({size // (1024 * 1024)} MB received so far)."
                            ),
                        )
                    if len(buf_for_magic) < magic_probe_bytes:
                        buf_for_magic.extend(chunk[: magic_probe_bytes - len(buf_for_magic)])
                    out.write(chunk)
        except HTTPException:
            raise
        except Exception as exc:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Failed to write '{name}': {exc}") from exc

        if is_pdf_ext:
            kind = filetype.guess(bytes(buf_for_magic))
            if kind is None or kind.mime != "application/pdf":
                dest_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=422,
                    detail=f"'{name}' does not appear to be a valid PDF (magic bytes mismatch).",
                )

        # Stale-derivative cleanup: the user has replaced the source PDF, so
        # every artifact that referenced the old content (converted MDs,
        # in-flight checkpoints) is no longer trustworthy — cached pages
        # would be misleading on a re-conversion.  Done only after the new
        # PDF is fully written AND validated so a rejected upload preserves
        # the previous state.
        if was_overwrite:
            stem = _stem(name)
            for md_file in _md_files_for_stem(self._mds_dir, stem):
                try:
                    md_file.unlink()
                except OSError as exc:
                    logger.warning(
                        "Failed to remove stale MD '%s' after overwrite: %s",
                        md_file, exc,
                    )
            self._discard_derived_caches(stem)
            logger.info(
                "Overwrote '%s' — wiped stale derivatives for stem '%s'",
                name, stem,
                extra={"operation": "upload", "file_name": name},
            )

        logger.info(
            "Uploaded '%s' (%d KB)",
            name,
            size // 1024,
            extra={"operation": "upload", "file_name": name},
        )

    def upload_files(self, files: list[UploadFile]) -> MultiUploadResponse:
        results: list[UploadFileResult] = []

        for file in files:
            name = file.filename or ""
            try:
                self.upload_file(file)
                results.append(UploadFileResult(filename=name, success=True, message="Uploaded successfully"))
            except HTTPException as exc:
                results.append(UploadFileResult(filename=name, success=False, message=exc.detail))
            except Exception as exc:
                results.append(UploadFileResult(filename=name, success=False, message=str(exc)))

        uploaded = sum(1 for r in results if r.success)
        return MultiUploadResponse(
            uploaded=uploaded,
            failed=len(results) - uploaded,
            results=results,
        )

    # ------------------------------------------------------------------
    # Convert
    # ------------------------------------------------------------------

    def convert_to_markdown(
        self,
        filename: str,
        converter_type: ConverterType = ConverterType.pymupdf,
        vlm_settings: VLMSettings | None = None,
        cloud_settings: CloudSettings | None = None,
        stop_event: threading.Event | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> ConvertResponse:
        """Convert a stored PDF to Markdown and persist the result."""
        filename = safe_filename(filename, "PDF filename")
        pdf_path = self._pdfs_dir / filename
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF '{filename}' not found")

        settings = get_settings()

        page_count: int | None = None
        try:
            import fitz
            with fitz.open(str(pdf_path)) as doc:
                page_count = doc.page_count
            if settings.MAX_PAGE_COUNT > 0 and page_count > settings.MAX_PAGE_COUNT:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"'{filename}' has {page_count} pages, which exceeds the "
                        f"configured limit of {settings.MAX_PAGE_COUNT}."
                    ),
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning(
                "Could not read page count for '%s': %s", filename, exc, exc_info=True
            )

        def _progress_handler(current: int, total: int) -> None:
            if on_progress:
                on_progress(current, total)
            if stop_event and stop_event.is_set():
                raise InterruptedError("Conversion cancelled by client disconnect")

        # Build the per-converter Markdown filename now so we can short-circuit
        # if the same document+converter has already been converted before.
        stem = _stem(filename)
        converter_token = _normalise_converter(converter_type.value)
        md_filename = f"{stem}_{converter_token}.md"
        md_path = self._mds_dir / md_filename

        # VLM-only: prepare the checkpoint store.  We snapshot the
        # "is there a partial conversion?" signal BEFORE honouring the
        # ``use_checkpoint`` opt-out — otherwise discarding the checkpoint
        # would erase the only evidence that the previous run left a partial
        # MD on disk, and the next branch would happily return that stale
        # partial Markdown instead of running a fresh conversion.
        checkpoint_store: CheckpointStore | None = None
        had_partial_checkpoint = False
        if converter_type == ConverterType.vlm and settings.VLM_CHECKPOINT_ENABLED:
            checkpoint_store = CheckpointStore(stem, self._mds_dir)
            had_partial_checkpoint = checkpoint_store.exists()
            use_checkpoint = vlm_settings.use_checkpoint if vlm_settings is not None else True
            if not use_checkpoint:
                checkpoint_store.discard()

        # A stored Markdown alone normally means "already converted, reuse" —
        # but for VLM, a checkpoint dir surviving the previous run signals
        # that the run completed with at least one failed page (we keep the
        # dir to enable "retry failed pages").  In that case we must re-run
        # so the cached failures get another attempt instead of returning a
        # stale partial result.  The check uses the pre-discard snapshot so
        # ``use_checkpoint=False`` still triggers a fresh conversion when
        # the previous run was partial.
        if md_path.exists() and not had_partial_checkpoint:
            logger.info(
                "Skipping conversion of '%s' with '%s' — '%s' already exists",
                filename, converter_type.value, md_filename,
                extra={"operation": "convert", "file_name": filename},
            )
            return ConvertResponse(
                success=True,
                md_filename=md_filename,
                message=f"'{md_filename}' already exists — reusing on-disk content",
                md_content=md_path.read_text(encoding="utf-8"),
            )

        logger.info(
            "Starting conversion of '%s' with converter '%s'",
            filename,
            converter_type.value,
            extra={"operation": "convert", "file_name": filename},
        )
        t0 = time.monotonic()

        # I/O-bound converters (VLM, Cloud) receive stop_event and on_progress.
        # CPU-bound converters run in isolated processes and cannot accept these.
        _is_io_bound = converter_type in (ConverterType.vlm, ConverterType.cloud)

        try:
            converter = _build_converter(
                converter_type,
                vlm_settings,
                cloud_settings,
                on_progress=_progress_handler if _is_io_bound else None,
                stop_event=stop_event if _is_io_bound else None,
                checkpoint_store=checkpoint_store,
            )
            md_content = converter.convert(pdf_path, total_pages=page_count)
        except InterruptedError:
            logger.info(
                "Conversion of '%s' was cancelled after %.0f ms",
                filename,
                (time.monotonic() - t0) * 1000,
                extra={"operation": "convert", "file_name": filename},
            )
            # Checkpoint is intentionally preserved on cancellation so the
            # next attempt can resume from the last persisted page.
            raise

        # ── Extract per-page outcome from the VLM converter ──────────────────
        # Other converters do not surface these attributes; default to None
        # so the response shape stays uniform but uninformative for them.
        failed_pages = getattr(converter, "failed_pages", None) or None
        resumed_pages = (
            getattr(converter, "resumed_from_pages", 0) if checkpoint_store else 0
        ) or None

        # Persist the final Markdown BEFORE deleting the checkpoint so a
        # failure between the write and the cleanup leaves the cache intact
        # for a retry — never the other way around.
        self._mds_dir.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_content, encoding="utf-8")

        # Clean up the checkpoint only when the conversion is fully clean
        # (no failed pages).  A partial conversion keeps its checkpoint so
        # "retry failed pages" — which re-submits the same convert request
        # — can resume the cached pages and re-attempt the failures.
        if checkpoint_store is not None and not failed_pages:
            checkpoint_store.discard()

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Conversion complete: '%s' → '%s' in %d ms (failed_pages=%s, resumed_pages=%s)",
            filename,
            md_filename,
            elapsed_ms,
            failed_pages,
            resumed_pages,
            extra={"operation": "convert", "file_name": filename, "duration_ms": elapsed_ms},
        )
        return ConvertResponse(
            success=True,
            md_filename=md_filename,
            message=f"Converted '{filename}' to Markdown using {converter_type.value}",
            md_content=md_content,
            failed_pages=failed_pages,
            resumed_pages=resumed_pages,
        )

    # ------------------------------------------------------------------
    # Convert MD → PDF
    # ------------------------------------------------------------------

    def convert_md_to_pdf(self, md_filename: str) -> MdToPdfResponse:
        from backend.scripts.md_to_pdf import _convert_file

        md_filename = safe_filename(md_filename, "Markdown filename")
        md_path = self._mds_dir / md_filename
        if not md_path.exists():
            raise HTTPException(status_code=404, detail=f"MD '{md_filename}' not found")

        self._pdfs_dir.mkdir(parents=True, exist_ok=True)
        pdf_filename = f"{_stem(md_filename)}.pdf"
        pdf_path = self._pdfs_dir / pdf_filename

        success = _convert_file(md_path, pdf_path)
        if not success:
            raise HTTPException(status_code=500, detail="MD to PDF conversion failed")

        return MdToPdfResponse(
            success=True,
            pdf_filename=pdf_filename,
            message=f"Converted '{md_filename}' to PDF",
        )

    def delete_document(self, filename: str) -> DeleteResponse:
        filename = safe_filename(filename, "document name")
        ext = Path(filename).suffix.lower()
        deleted: list[str] = []
        stem = _stem(filename)

        if ext == ".md":
            md_path = self._mds_dir / filename
            if not md_path.exists():
                raise HTTPException(status_code=404, detail=f"MD '{filename}' not found")
            md_path.unlink()
            deleted.append(str(md_path))

            chunks_path = self._chunks_dir / stem
            if chunks_path.exists():
                shutil.rmtree(chunks_path)
                deleted.append(str(chunks_path))

            # Same cascade cleanup the PDF branch performs.  Without this,
            # deleting a converted variant (e.g. ``report_vlm.md``) leaks its
            # enrichment-checkpoint dir, and deleting a bare-MD upload leaks
            # its document-summary file as well.  The discards are stem-glob
            # based so they only touch caches that belong to this MD: the
            # shared per-PDF summary survives a converted-variant delete.
            self._discard_derived_caches(stem)

            associated = len(deleted) - 1
            return DeleteResponse(
                success=True,
                deleted=deleted,
                message=f"Deleted '{filename}' and {associated} associated file(s)",
            )

        pdf_path = self._pdfs_dir / filename
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF '{filename}' not found")

        pdf_path.unlink()
        deleted.append(str(pdf_path))

        for md_file in _md_files_for_stem(self._mds_dir, stem):
            try:
                md_file.unlink()
                deleted.append(str(md_file))
            except OSError:
                pass

        # Drop any checkpoint directories belonging to this stem.  They would
        # otherwise become orphans referring to a PDF that no longer exists
        # and would silently feed stale page content into the next document
        # uploaded under the same name.  Both VLM and enrichment checkpoint
        # dirs live under ``mds/.checkpoints/`` and must be cleaned together.
        # Document summaries (``mds/{stem}_{converter}.summary.json``) follow
        # the same lifecycle and are cleaned here as well.
        self._discard_derived_caches(stem)

        chunks_path = self._chunks_dir / stem
        if chunks_path.exists():
            shutil.rmtree(chunks_path)
            deleted.append(str(chunks_path))

        associated = len(deleted) - 1
        return DeleteResponse(
            success=True,
            deleted=deleted,
            message=f"Deleted '{filename}' and {associated} associated file(s)",
        )

    def delete_documents(self, filenames: list[str]) -> DeleteResponse:
        all_deleted: list[str] = []
        errors: list[str] = []

        for filename in filenames:
            try:
                result = self.delete_document(filename)
                all_deleted.extend(result.deleted)
            except HTTPException as exc:
                errors.append(f"{filename}: {exc.detail}")
            except OSError as exc:
                errors.append(f"{filename}: {exc}")

        if errors and not all_deleted:
            raise HTTPException(status_code=404, detail="; ".join(errors))

        deleted_count = len(filenames) - len(errors)
        message = f"Deleted {deleted_count} document(s)"
        if errors:
            message += f"; {len(errors)} not found: {', '.join(errors)}"

        return DeleteResponse(
            success=len(errors) == 0,
            deleted=all_deleted,
            message=message,
        )
