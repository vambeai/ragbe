"""
Router for document management endpoints.

Prefix: /api

# VERSION 3 — Per-batch ProcessPoolExecutor for CPU-bound converters
# CPU-bound converters (PyMuPDF, Docling, MarkItDown) now run in a dedicated
# per-batch ProcessPoolExecutor so that cancellation can terminate worker
# processes cleanly without breaking a shared pool.
# VLM and Cloud converters continue to run via asyncio.to_thread (I/O-bound).
#
# Cancellation path:
#   1. stop_events are set for all I/O-bound (VLM/Cloud) conversions.
#   2. concurrent.futures.Future objects are cancelled for CPU-bound work.
#   3. Worker processes in the per-batch executor are terminated via SIGTERM.
#   4. The per-batch executor is shut down; the OS reaps the workers.

Conversion streams progress via Server-Sent Events (SSE):

    POST /api/convert
        Accepts one or more filenames.  Runs up to MAX_CONCURRENT_CONVERSIONS
        in parallel; remaining files are queued server-side.

        SSE event types (consistent for 1 or N files):
            {"type": "file_start",    "filename": "...", "index": 1, "total": N}
            {"type": "progress",      "filename": "...", "current": 3, "total": 10, "percentage": 30}
                -- VLM/Cloud only; emitted after each page / after API responds
            {"type": "file_done",     "filename": "...", "success": true,
             "md_filename": "...", "md_content": "..."}
            {"type": "file_done",     "filename": "...", "success": false, "error": "..."}
            {"type": "file_progress", "filename": "...", "current": 1, "total": N, "percentage": 33}
                -- emitted after every file completes (success or failure)
            {"type": "batch_done",    "succeeded": N, "failed": M}
            {"type": "error",         "status": 4xx/5xx, "message": "..."}
            {"type": "cancelled"}
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import AsyncGenerator

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from backend.config import get_settings
from backend.models.schemas import (
    CheckpointInfoResponse,
    ConvertRequest,
    ConvertResponse,
    ConverterType,
    DeleteResponse,
    DocumentInfo,
    MarkdownContentResponse,
    MarkdownVersionsResponse,
    MdToPdfResponse,
    MultiUploadResponse,
)
from backend.services.document_service import (
    DocumentService,
    _init_cpu_worker,
    convert_in_process,
)
from backend.utils.executor import cancel_cpu_executor
from backend.utils.sse import (
    run_sse_event_loop,
    sse_event as _sse,
)

router = APIRouter(prefix="/api", tags=["documents"])
_svc = DocumentService()
logger = logging.getLogger(__name__)


# Converters that run in a per-batch ProcessPoolExecutor (CPU-bound).
# VLM and Cloud are excluded — they are I/O-bound (HTTP calls) and run in threads.
_CPU_BOUND_CONVERTERS = frozenset({
    ConverterType.pymupdf,
    ConverterType.docling,
    ConverterType.markitdown,
})


# ── Read endpoints ────────────────────────────────────────────────────────────

@router.get("/documents", response_model=list[str])
async def list_documents():
    """Return a sorted list of all available document filenames."""
    return await asyncio.to_thread(_svc.list_documents)


@router.get("/documents/metadata")
async def list_documents_metadata():
    """Return metadata (including has_markdown) for every document."""
    return await asyncio.to_thread(_svc.list_documents_metadata)


@router.get("/document/{filename}", response_model=DocumentInfo)
async def get_document(filename: str):
    """Return metadata and existing Markdown content for a document."""
    return await asyncio.to_thread(_svc.get_document, filename)


@router.get("/pdf/{filename}")
async def serve_pdf(filename: str):
    """Serve a PDF file for inline viewing or download."""
    pdf_path = _svc.get_pdf_path(filename)
    return FileResponse(pdf_path, media_type="application/pdf", filename=filename)


# ── Versioned Markdown listing / fetch ────────────────────────────────────────

@router.get(
    "/documents/{document_name}/markdowns",
    response_model=MarkdownVersionsResponse,
)
async def list_markdown_versions(document_name: str):
    """Return every available Markdown version for a document.

    Each entry distinguishes between converted variants
    (``source="converted"``, with the parsed converter token) and uploaded
    files (``source="uploaded"``, ``converter=null``).  This is the data
    backing the frontend's Markdown-version dropdown.
    """
    versions = await asyncio.to_thread(_svc.list_markdown_versions, document_name)
    return MarkdownVersionsResponse(document_name=document_name, versions=versions)


@router.get(
    "/documents/{document_name}/markdowns/{identifier}",
    response_model=MarkdownContentResponse,
)
async def get_markdown_version(document_name: str, identifier: str):
    """Return the full text of one specific Markdown version.

    ``identifier`` may be either a converter name (e.g. ``pymupdf4llm``,
    ``docling``) or the original filename (for uploaded MDs whose name
    does not match the converter pattern).
    """
    return await asyncio.to_thread(_svc.get_markdown_content, document_name, identifier)


# ── VLM checkpoint inspection ─────────────────────────────────────────────────

@router.get(
    "/documents/{document_name}/checkpoint",
    response_model=CheckpointInfoResponse,
)
async def get_vlm_checkpoint(document_name: str):
    """Return the VLM checkpoint state for a document.

    The frontend calls this before kicking off a VLM conversion to decide
    whether to surface a "Resume available" indicator alongside the
    progress modal.  Reports ``exists=false`` (and an empty
    ``completed_pages`` list) when no checkpoint is on disk.
    """
    info = await asyncio.to_thread(_svc.get_vlm_checkpoint_info, document_name)
    return CheckpointInfoResponse(**info)


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=MultiUploadResponse)
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload one or more PDF / Markdown files."""
    return await asyncio.to_thread(_svc.upload_files, files)


# ── Unified conversion endpoint (SSE) ────────────────────────────────────────

@router.post("/convert")
async def convert_pdfs(
    http_request: Request,
    request: ConvertRequest,
):
    """Convert one or more PDFs to Markdown, streaming progress via SSE."""

    async def event_stream() -> AsyncGenerator[str, None]:
        semaphore = http_request.app.state.conversion_semaphore
        filenames = request.filenames
        total = len(filenames)
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        succeeded = 0
        failed = 0
        _lock = asyncio.Lock()

        _stop_events: list[threading.Event] = []
        _stop_events_lock = asyncio.Lock()

        _settings = get_settings()
        watchdog_s = _settings.SSE_WATCHDOG_TIMEOUT_S
        queue_timeout_s = _settings.SSE_QUEUE_GET_TIMEOUT_S
        cancel_wait_s = _settings.SSE_CANCEL_WAIT_TIMEOUT_S
        loop = asyncio.get_running_loop()
        is_cpu_bound = request.converter in _CPU_BOUND_CONVERTERS

        # concurrent.futures.Future objects for in-flight CPU-bound jobs.
        # Tracked so _cancel_all() can call .cancel() on queued-but-not-started
        # futures.  Already-running futures cannot be cancelled without
        # terminating the worker process, which would break the shared pool;
        # those jobs run to completion and their results are simply discarded.
        _cpu_futures: list = []

        if is_cpu_bound:
            _shared_executor = http_request.app.state.cpu_converter_executor

            async def _dispatch(fn: str, _stop, _on_progress) -> ConvertResponse:
                cf = _shared_executor.submit(convert_in_process, fn, request.converter)
                _cpu_futures.append(cf)
                try:
                    return await asyncio.wrap_future(cf)
                finally:
                    try:
                        _cpu_futures.remove(cf)
                    except ValueError:
                        pass
        else:
            async def _dispatch(fn: str, _stop, _on_progress) -> ConvertResponse:
                return await asyncio.to_thread(
                    _svc.convert_to_markdown,
                    fn,
                    converter_type=request.converter,
                    vlm_settings=request.vlm,
                    cloud_settings=request.cloud,
                    stop_event=_stop,
                    on_progress=_on_progress,
                )

        async def convert_one(idx: int, fn: str) -> None:
            nonlocal succeeded, failed

            # stop_event is only meaningful for I/O-bound converters (VLM/Cloud).
            # CPU-bound converters run in isolated processes and cannot receive
            # a threading.Event across the process boundary.
            stop = threading.Event() if not is_cpu_bound else None
            if stop is not None:
                async with _stop_events_lock:
                    _stop_events.append(stop)

            async with semaphore:
                if await http_request.is_disconnected():
                    return

                await queue.put({"type": "file_start", "filename": fn, "index": idx + 1, "total": total})

                def _on_progress(current: int, total_pages: int) -> None:
                    try:
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {
                                "type": "progress",
                                "filename": fn,
                                "current": current,
                                "total": total_pages,
                                "file_index": idx + 1,
                                "file_total": total,
                                "percentage": round(current / total_pages * 100) if total_pages else 0,
                            },
                        )
                    except Exception as _err:
                        logger.warning("Failed to queue progress event for '%s': %s", fn, _err)

                t0 = time.monotonic()
                _done = 0
                try:
                    result = await _dispatch(fn, stop, _on_progress)

                    async with _lock:
                        succeeded += 1
                        _done = succeeded + failed
                    # Use put_nowait (no await) for both events so no other
                    # coroutine can interleave between them and produce
                    # out-of-order file_progress percentages on the client.
                    queue.put_nowait({
                        "type": "file_done",
                        "filename": fn,
                        "success": True,
                        "md_filename": result.md_filename,
                        "md_content": result.md_content,
                        "duration_ms": int((time.monotonic() - t0) * 1000),
                        "failed_pages": result.failed_pages,
                        "resumed_pages": result.resumed_pages,
                    })
                except Exception as exc:
                    async with _lock:
                        failed += 1
                        _done = succeeded + failed
                    error_summary = f"{type(exc).__name__}: {str(exc)[:120]}"
                    queue.put_nowait({"type": "file_done", "filename": fn, "success": False, "error": error_summary})
                    logger.warning(
                        "Convert failed for '%s': %s",
                        fn,
                        exc,
                        exc_info=True,
                        extra={"operation": "convert", "file_name": fn},
                    )

                queue.put_nowait({
                    "type": "file_progress",
                    "filename": fn,
                    "current": _done,
                    "total": total,
                    "percentage": round(_done / total * 100),
                })

        async def run_all() -> None:
            tasks = [asyncio.create_task(convert_one(i, fn)) for i, fn in enumerate(filenames)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                    logger.error(
                        "Unexpected exception from conversion task: %s",
                        res,
                        exc_info=res,
                        extra={"operation": "convert"},
                    )
            await queue.put(None)  # sentinel

        runner = asyncio.create_task(run_all())

        async def _cancel_all() -> None:
            # 1. Signal I/O-bound converters (VLM / Cloud) to stop.
            async with _stop_events_lock:
                for se in _stop_events:
                    se.set()

            # 2. Cancel CPU-bound futures; SIGTERM workers; swap executor.
            if is_cpu_bound:
                s = get_settings()
                await cancel_cpu_executor(
                    _cpu_futures,
                    http_request.app.state,
                    "cpu_converter_executor",
                    s.MAX_CONCURRENT_CONVERSIONS,
                    _init_cpu_worker,
                    "converter worker",
                    logger,
                )

            # 3. Cancel the asyncio runner task.
            runner.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(runner), timeout=cancel_wait_s)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        async def _safe_cancel() -> None:
            # Idempotent wrapper: the executor swap inside _cancel_all is
            # heavy work; skip it once the runner has finished cleanly so
            # the finally-block safety net stays a no-op on success.
            if not runner.done():
                await _cancel_all()

        def _handle(event: dict) -> tuple[str, bool]:
            return _sse(event), False

        def _on_complete():
            return [_sse({"type": "batch_done", "succeeded": succeeded, "failed": failed})]

        async for frame in run_sse_event_loop(
            queue=queue,
            http_request=http_request,
            on_cancel=_safe_cancel,
            handle_event=_handle,
            watchdog_s=watchdog_s,
            queue_timeout_s=queue_timeout_s,
            log_name=f"conversion ({total} doc(s))",
            on_complete=_on_complete,
        ):
            yield frame

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── MD → PDF conversion ───────────────────────────────────────────────────────

@router.post("/md-to-pdf/{filename}", response_model=MdToPdfResponse)
async def convert_md_to_pdf(filename: str, http_request: Request):
    """Convert a stored Markdown file to PDF using markdown-pdf."""
    try:
        return await asyncio.to_thread(_svc.convert_md_to_pdf, filename)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error in MD→PDF conversion of '%s'", filename)
        raise HTTPException(status_code=500, detail="MD to PDF conversion failed due to an internal error")
# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/documents", response_model=DeleteResponse)
async def delete_documents(filenames: list[str]):
    """Delete one or more documents and all their derived files."""
    return await asyncio.to_thread(_svc.delete_documents, filenames)
