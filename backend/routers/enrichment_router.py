"""
Router for LLM enrichment endpoints.

Prefix: /api/enrich

Two endpoints stream progress via Server-Sent Events (SSE):

    POST /api/enrich/markdown/pipeline
        Runs the full enrichment pipeline (regex cleanup, structure-aware
        split, per-piece LLM correction with rolling context, per-piece
        checkpoint) on a stored Markdown file.
        SSE event types:
            {"type": "start",        "operation": "enrich_pipeline"}
            {"type": "cleanup_done", "report": {...}}
            {"type": "split_done",   "pieces": N}
            {"type": "piece_start",  "index": i, "total": N}
            {"type": "piece_done",   "index": i, "total": N,
             "cached": bool, "succeeded": bool, "error": "..."?}
            {"type": "done",         "enriched_content": "...", "stats": {...}}
            {"type": "error",        "status": 4xx/5xx, "message": "..."}
            {"type": "cancelled"}

    POST /api/enrich/chunks
        SSE event types:
            {"type": "start",      "operation": "enrich_chunks", "total": N}
            {"type": "chunk_done", "operation": "enrich_chunks",
             "current": 1, "total": N, "percentage": 50, "chunk": {...enriched fields...}}
            {"type": "chunk_error","operation": "enrich_chunks",
             "current": 1, "total": N, "chunk_index": 0, "message": "..."}
            {"type": "done",       "operation": "enrich_chunks", "total_chunks": N}
            {"type": "error",      "status": 4xx/5xx, "message": "..."}
            {"type": "cancelled"}
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

from backend.config import get_settings
from backend.models.schemas import (
    DocumentSummaryPayload,
    DocumentSummaryResponse,
    DocumentSummaryStatus,
    EnrichChunksRequest,
    EnrichPipelineRequest,
    SummaryGenerateRequest,
    SummaryUpdateRequest,
)
from backend.services.document_service import DocumentService
from backend.services.document_summary import (
    DocumentSummary,
    DocumentSummaryStore,
    StoredSummary,
    get_or_generate_summary,
    save_user_edit,
)
from backend.services.enrichment_checkpoint import EnrichmentCheckpointStore
from backend.services.enrichment_pipeline import run_enrichment_pipeline
from backend.services.enrichment_service import EnrichmentService
from backend.utils.markdown import clean_markdown, split_markdown
from backend.utils.naming import doc_stem_from_md
from backend.utils.path import safe_filename
from backend.utils.sse import (
    run_sse_event_loop,
    sse_error as _sse_error,
    sse_event as _sse,
)

router = APIRouter(prefix="/api/enrich", tags=["enrichment"])
_doc_svc = DocumentService()



@router.post("/chunks")
async def enrich_chunks(http_request: Request, body: EnrichChunksRequest):
    """Enrich an array of chunks with LLM-generated metadata, streaming per-chunk events.

    Chunks are processed concurrently (up to ENRICHMENT_MAX_CONCURRENT_CHUNKS in flight
    at once). A ``chunk_done`` event is emitted after each chunk completes so the
    frontend can update the UI incrementally.

    Per-chunk errors are reported as ``chunk_error`` events and do not abort the
    remaining chunks.  A timed-out chunk (after retries) is reported as
    ``chunk_error`` and processing continues with the next chunk.

    A keepalive heartbeat comment is emitted every 30 s during idle periods so
    the frontend's connection-lost watchdog does not fire between slow chunks.
    A backend watchdog cancels all in-flight chunks if no event arrives for
    SSE_WATCHDOG_TIMEOUT_S seconds.

    SSE events: ``start`` → ``chunk_done`` × N → ``done``
    (or ``chunk_error`` for individual failures, ``error`` for fatal errors,
    ``cancelled`` on client disconnect).
    """

    async def event_stream() -> AsyncGenerator[str, None]:
        if await http_request.is_disconnected():
            yield _sse({"type": "cancelled"})
            return

        http_client = http_request.app.state.http_client_async
        s = body.settings
        svc = EnrichmentService(
            model=s.model, base_url=s.base_url, api_key=s.api_key,
            temperature=s.temperature, user_prompt=s.user_prompt, http_client=http_client,
        )
        chunks = [c.model_dump() for c in body.chunks]
        total = len(chunks)

        if total == 0:
            yield _sse({"type": "start", "operation": "enrich_chunks", "total": 0})
            yield _sse({"type": "done", "operation": "enrich_chunks", "total_chunks": 0, "succeeded": 0})
            return

        # Silent summary attachment (Phase B): if the caller passed
        # ``md_filename`` AND a per-PDF summary exists on disk, attach
        # it to every chunk-enrichment prompt as document-level context.
        # Chunk enrichment never generates a summary on its own — the
        # user owns that decision via the markdown enrichment flow's
        # review modal.  Failures here degrade silently (chunks proceed
        # without context); never abort the whole batch.
        document_summary = None
        if body.md_filename:
            try:
                md_name = safe_filename(body.md_filename, "Markdown filename")
                doc_stem = doc_stem_from_md(md_name)
                stored = DocumentSummaryStore(doc_stem, _doc_svc._mds_dir).load()
                if stored is not None and not stored.summary.is_empty():
                    document_summary = stored.summary
            except Exception as exc:  # noqa: BLE001 — silent degradation
                logger.warning(
                    "Chunk enrichment: failed to load summary for %r — continuing without: %s",
                    body.md_filename, exc,
                )

        _settings = get_settings()
        watchdog_s = _settings.SSE_WATCHDOG_TIMEOUT_S
        queue_timeout_s = _settings.SSE_QUEUE_GET_TIMEOUT_S
        # Use the global semaphore from app.state so the cap is enforced across
        # ALL concurrent /enrich/chunks requests, not just within this one call.
        semaphore = http_request.app.state.enrichment_chunks_semaphore

        yield _sse({"type": "start", "operation": "enrich_chunks", "total": total})

        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def enrich_one(chunk: dict) -> None:
            chunk_index = chunk["index"]
            content = chunk.get("content", "")
            async with semaphore:
                try:
                    enriched = await svc.enrich_chunk(content, document_summary=document_summary)
                    _kw = enriched.get("keywords", [])
                    _qs = enriched.get("questions", [])
                    result = {
                        "index": chunk_index,
                        "content": content,
                        "cleaned_chunk": enriched.get("cleaned_chunk", ""),
                        "title": enriched.get("title", ""),
                        "context": enriched.get("context", ""),
                        "summary": enriched.get("summary", ""),
                        "keywords": _kw if isinstance(_kw, list) else [],
                        "questions": _qs if isinstance(_qs, list) else [],
                        "metadata": chunk.get("metadata", {}),
                        "start": chunk.get("start", 0),
                        "end": chunk.get("end", 0),
                    }
                    await queue.put({"type": "chunk_done", "chunk": result, "chunk_index": chunk_index})
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error(
                        "Chunk enrichment failed at index %s (will continue): %s",
                        chunk_index,
                        exc,
                    )
                    await queue.put({"type": "chunk_error", "chunk_index": chunk_index,
                                     "message": "Chunk enrichment failed. Check server logs."})

        async def run_all() -> None:
            tasks = [asyncio.create_task(enrich_one(c)) for c in chunks]
            await asyncio.gather(*tasks, return_exceptions=True)
            await queue.put(None)  # sentinel

        runner = asyncio.create_task(run_all())

        completed = 0
        succeeded = 0

        async def _cancel() -> None:
            if not runner.done():
                runner.cancel()
                await asyncio.gather(runner, return_exceptions=True)

        def _handle(event: dict) -> tuple[str, bool]:
            nonlocal completed, succeeded
            completed += 1
            if event["type"] == "chunk_done":
                succeeded += 1
                return _sse({
                    "type": "chunk_done",
                    "operation": "enrich_chunks",
                    "current": completed,
                    "total": total,
                    "percentage": round(completed / total * 100),
                    "chunk": event["chunk"],
                }), False
            return _sse({
                "type": "chunk_error",
                "operation": "enrich_chunks",
                "current": completed,
                "total": total,
                "chunk_index": event["chunk_index"],
                "message": event["message"],
            }), False

        def _on_complete():
            return [_sse({
                "type": "done",
                "operation": "enrich_chunks",
                "total_chunks": total,
                "succeeded": succeeded,
            })]

        async for frame in run_sse_event_loop(
            queue=queue,
            http_request=http_request,
            on_cancel=_cancel,
            handle_event=_handle,
            watchdog_s=watchdog_s,
            queue_timeout_s=queue_timeout_s,
            log_name="chunk enrichment",
            on_complete=_on_complete,
        ):
            yield frame

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Pipeline endpoint (regex + structure-aware split + per-piece LLM) ────────

@router.post("/markdown/pipeline")
async def enrich_markdown_pipeline(http_request: Request, body: EnrichPipelineRequest):
    """Run the full enrichment pipeline on a stored Markdown file.

    Differences from ``/markdown``:

        * Pre-cleans the source with deterministic regex passes before any
          LLM call (page numbers, repeated headers/footers, mid-word
          hyphen wraps, blank-line collapse).
        * Splits the cleaned source at structural boundaries (headings,
          paragraphs) without ever cutting tables, code blocks, or lists.
        * Runs the LLM on each piece with rolling context from the
          *original* preceding piece (not the corrected one) to avoid
          drift.
        * Persists each successful piece to a checkpoint dir; a re-run
          with identical inputs is essentially free.
        * Per-piece failures fall back to the original post-cleanup
          content; they do not abort the run.

    The document-level summary is NOT built here — the frontend's
    review modal calls ``/summary/generate`` first so the user can
    inspect and edit before any piece correction runs.  Whichever
    summary is on disk at request time is loaded and attached to every
    piece prompt; if no summary exists the pipeline runs without
    document-level context.

    SSE event sequence (one cleanup_done + one split_done per request;
    piece_start / piece_done emitted per piece):

        {"type": "start"}
        {"type": "cleanup_done",      "report": {...}}
        {"type": "split_done",        "pieces": N}
        {"type": "piece_start",       "index": i, "total": N}
        {"type": "piece_done",        "index": i, "total": N, "succeeded": bool, "cached": bool}
          ... (one per piece correction) ...
        {"type": "done", "enriched_content": "...", "stats": {...}}
        OR
        {"type": "error", "status": 5xx, "message": "..."}
        OR
        {"type": "cancelled"}
    """

    async def event_stream() -> AsyncGenerator[str, None]:
        if await http_request.is_disconnected():
            yield _sse({"type": "cancelled"})
            return

        # Validate filename + load source content up-front so file errors
        # surface before we start the LLM dance.
        try:
            md_name = safe_filename(body.filename, "Markdown filename")
        except HTTPException as exc:
            yield _sse_error(exc.status_code, exc.detail)
            return

        md_path = _doc_svc._mds_dir / md_name
        if not md_path.exists():
            yield _sse_error(404, f"Markdown file '{md_name}' not found")
            return

        try:
            source_markdown = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.exception("Failed to read source markdown for pipeline")
            yield _sse_error(500, f"Could not read '{md_name}': {exc}")
            return

        if not source_markdown.strip():
            yield _sse_error(422, f"'{md_name}' is empty — nothing to enrich")
            return

        yield _sse({"type": "start", "operation": "enrich_pipeline"})

        # Checkpoint store keyed by markdown stem (i.e. per converter
        # variant — corrections are not portable across converters).
        # ``use_checkpoint=False`` wipes any stale cache before starting
        # so a "rerun from scratch" toggle in the UI behaves predictably.
        from pathlib import Path as _Path
        stem = _Path(md_name).stem
        store = EnrichmentCheckpointStore(stem, _doc_svc._mds_dir)
        if not body.use_checkpoint:
            store.discard()

        # Document summary, in contrast, is keyed by the SOURCE PDF — so
        # switching converters does not invalidate it.  The pipeline no
        # longer generates the summary itself; the frontend's review
        # modal calls ``/summary/generate`` first so the user can
        # inspect and edit the summary before any piece-correction LLM
        # calls fire.  Here we just load whatever is cached and hand it
        # to the pipeline; absence is fine (pipeline runs without a
        # document-level context block in that case).
        #
        # ``body.use_summary`` lets the caller force the no-summary path
        # even when a record exists on disk — this is what makes the
        # modal's Skip button and the Settings → "Skip document summary"
        # toggle actually do something when the user already has a
        # cached summary they want to bypass for this one run.
        doc_stem = doc_stem_from_md(md_name)
        document_summary = None
        if body.use_summary:
            summary_store = DocumentSummaryStore(doc_stem, _doc_svc._mds_dir)
            loaded_summary = summary_store.load()
            document_summary = loaded_summary.summary if loaded_summary is not None else None

        http_client = http_request.app.state.http_client_async
        s = body.settings
        svc = EnrichmentService(
            model=s.model, base_url=s.base_url, api_key=s.api_key,
            temperature=s.temperature, user_prompt=s.user_prompt, http_client=http_client,
        )

        _settings = get_settings()
        watchdog_s = _settings.SSE_WATCHDOG_TIMEOUT_S
        queue_timeout_s = _settings.SSE_QUEUE_GET_TIMEOUT_S
        concurrency = _settings.ENRICHMENT_MAX_CONCURRENT_CHUNKS

        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        cancelled_flag = {"v": False}

        def _on_progress(event: dict) -> None:
            # Bridge sync pipeline events → async queue for the SSE loop.
            # call_soon_threadsafe isn't needed because the pipeline runs in
            # the same loop, but using put_nowait keeps the path non-blocking
            # if events arrive faster than the consumer can yield them.
            try:
                queue.put_nowait(event)
            except Exception as _exc:
                logger.warning("Pipeline progress queue full: dropping event: %s", _exc)

        def _stop_check() -> bool:
            return cancelled_flag["v"]

        async def run_pipeline() -> None:
            try:
                result = await run_enrichment_pipeline(
                    source_markdown=source_markdown,
                    service=svc,
                    checkpoint_store=store,
                    document_summary=document_summary,
                    on_progress=_on_progress,
                    stop_check=_stop_check,
                    concurrency=concurrency,
                )
                await queue.put({
                    "type": "done",
                    "enriched_content": result.corrected_markdown,
                    "stats": {
                        "pieces": result.pieces,
                        "cached_pieces": result.cached_pieces,
                        "failed_pieces": result.failed_pieces,
                        "cleanup": result.cleanup_report.as_dict(),
                        "summary": {
                            "present": result.document_summary is not None,
                        },
                    },
                })
            except (asyncio.CancelledError, InterruptedError):
                await queue.put({"type": "cancelled"})
            except HTTPException as exc:
                await queue.put({"type": "error", "status": exc.status_code, "message": exc.detail})
            except Exception:
                logger.exception("Unexpected error during pipeline enrichment")
                await queue.put({"type": "error", "status": 500,
                                 "message": "An internal error occurred. Check server logs."})
            finally:
                await queue.put(None)  # sentinel

        runner = asyncio.create_task(run_pipeline())

        async def _cancel() -> None:
            cancelled_flag["v"] = True
            if not runner.done():
                runner.cancel()
                await asyncio.gather(runner, return_exceptions=True)

        def _handle(event: dict) -> tuple[str, bool]:
            etype = event.get("type")
            if etype == "error":
                return _sse_error(event["status"], event["message"]), True
            if etype == "cancelled":
                return _sse({"type": "cancelled"}), True
            # All other event types (including ``done``) pass through verbatim;
            # ``done`` is terminal so we also stop the loop.
            return _sse(event), etype == "done"

        async for frame in run_sse_event_loop(
            queue=queue,
            http_request=http_request,
            on_cancel=_cancel,
            handle_event=_handle,
            watchdog_s=watchdog_s,
            queue_timeout_s=queue_timeout_s,
            log_name="pipeline enrichment",
        ):
            yield frame

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Document-summary endpoints ──────────────────────────────────────────────
#
# The summary lives at ``{mds_dir}/{doc_stem}.summary.json`` — one record per
# source PDF, shared across converter variants.  These three endpoints let
# the frontend's Summary Review modal load, regenerate, and edit the record
# before the enrichment pipeline ever runs.  Pipeline endpoints only read
# the stored summary; they never produce or modify it.


def _payload_to_document_summary(payload: DocumentSummaryPayload) -> DocumentSummary:
    """Convert the API-shaped payload into the in-memory dataclass."""
    return DocumentSummary(
        topic=payload.topic.strip(),
        narrative=payload.narrative.strip(),
    )


def _stored_to_response(filename: str, doc_stem: str, stored: StoredSummary) -> DocumentSummaryResponse:
    """Materialise a :class:`StoredSummary` into the wire response."""
    return DocumentSummaryResponse(
        filename=filename,
        doc_stem=doc_stem,
        summary=DocumentSummaryPayload(
            topic=stored.summary.topic,
            narrative=stored.summary.narrative,
        ),
        status=DocumentSummaryStatus(
            user_edited=stored.user_edited,
            generated_at=stored.generated_at,
        ),
    )


@router.get("/summary", response_model=DocumentSummaryResponse)
async def get_summary(filename: str):
    """Return the cached document summary for the markdown variant.

    Resolves the per-PDF stem from the markdown filename and reads the
    on-disk record.  Returns 404 when no summary has been generated
    yet — the frontend handles that by offering the user a "Generate"
    button in the review modal.
    """
    try:
        md_name = safe_filename(filename, "Markdown filename")
    except HTTPException:
        raise
    doc_stem = doc_stem_from_md(md_name)
    store = DocumentSummaryStore(doc_stem, _doc_svc._mds_dir)
    stored = store.load()
    if stored is None:
        raise HTTPException(status_code=404, detail=f"No summary cached for '{md_name}'")
    return _stored_to_response(md_name, doc_stem, stored)


@router.put("/summary", response_model=DocumentSummaryResponse)
async def put_summary(body: SummaryUpdateRequest):
    """Persist a user-edited summary.

    Sets ``user_edited=True`` so future enrichment runs treat the edited
    content as authoritative.  Preserves the existing piece-extraction
    cache and source-hash metadata so a later Regenerate can still
    benefit from per-piece reuse.
    """
    try:
        md_name = safe_filename(body.filename, "Markdown filename")
    except HTTPException:
        raise
    doc_stem = doc_stem_from_md(md_name)
    edited = _payload_to_document_summary(body.summary)
    stored = save_user_edit(
        doc_stem=doc_stem,
        mds_dir=_doc_svc._mds_dir,
        edited_summary=edited,
    )
    return _stored_to_response(md_name, doc_stem, stored)


@router.post("/summary/generate")
async def generate_summary(http_request: Request, body: SummaryGenerateRequest):
    """Run the summary stages (cleanup → split → extract → reduce) and
    return the resulting :class:`DocumentSummaryResponse` over SSE.

    ``force=True`` discards any cached record before generation; this is
    the path the review modal's Regenerate button uses.  When ``force``
    is false and a cached record exists, this endpoint short-circuits
    and emits a single ``summary_ready`` (cached=True) followed by
    ``done`` — no LLM calls.

    SSE event sequence::

        {"type": "start"}
        {"type": "cleanup_done",     "report": {...}}
        {"type": "split_done",       "pieces": N}
        {"type": "summary_progress", "current": i, "total": N, "cached": bool}
          ...
        {"type": "summary_ready",    "cached": bool, "extractions": N,
                                     "failed": N, "reduce_failed": bool}
        {"type": "done", "summary": DocumentSummaryResponse}
    """

    async def event_stream() -> AsyncGenerator[str, None]:
        if await http_request.is_disconnected():
            yield _sse({"type": "cancelled"})
            return

        try:
            md_name = safe_filename(body.filename, "Markdown filename")
        except HTTPException as exc:
            yield _sse_error(exc.status_code, exc.detail)
            return

        md_path = _doc_svc._mds_dir / md_name
        if not md_path.exists():
            yield _sse_error(404, f"Markdown file '{md_name}' not found")
            return

        try:
            source_markdown = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.exception("Failed to read source markdown for summary generation")
            yield _sse_error(500, f"Could not read '{md_name}': {exc}")
            return

        if not source_markdown.strip():
            yield _sse_error(422, f"'{md_name}' is empty — nothing to summarise")
            return

        doc_stem = doc_stem_from_md(md_name)
        pdf_candidate = _doc_svc._pdfs_dir / f"{doc_stem}.pdf"
        pdf_path = pdf_candidate if pdf_candidate.is_file() else None

        yield _sse({"type": "start", "operation": "generate_summary"})

        # Wipe stale record up-front when forcing regeneration so the
        # builder doesn't try to reuse prior extractions.  The builder
        # itself handles ``force_regenerate`` semantics on cache hit;
        # we discard here so user-edited summaries are also overwritten
        # on Regenerate (their ``user_edited`` flag would otherwise win).
        store = DocumentSummaryStore(doc_stem, _doc_svc._mds_dir)
        if body.force:
            store.discard()

        http_client = http_request.app.state.http_client_async
        s = body.settings
        svc = EnrichmentService(
            model=s.model, base_url=s.base_url, api_key=s.api_key,
            temperature=s.temperature, user_prompt=s.user_prompt, http_client=http_client,
        )

        _settings = get_settings()
        watchdog_s = _settings.SSE_WATCHDOG_TIMEOUT_S
        queue_timeout_s = _settings.SSE_QUEUE_GET_TIMEOUT_S
        concurrency = _settings.ENRICHMENT_MAX_CONCURRENT_CHUNKS

        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        cancelled_flag = {"v": False}

        def _on_progress(event: dict) -> None:
            try:
                queue.put_nowait(event)
            except Exception as _exc:
                logger.warning("Summary progress queue full: dropping event: %s", _exc)

        def _stop_check() -> bool:
            return cancelled_flag["v"]

        async def run_summary() -> None:
            try:
                # Pre-cleanup + split run here so the builder receives
                # the same pieces the pipeline would.  Both are pure
                # functions (no I/O) and cheap.
                cleaned, report = clean_markdown(source_markdown)
                _on_progress({"type": "cleanup_done", "report": report.as_dict()})
                pieces = split_markdown(cleaned)
                _on_progress({"type": "split_done", "pieces": len(pieces)})

                await get_or_generate_summary(
                    pieces=pieces,
                    cleaned_source=cleaned,
                    doc_stem=doc_stem,
                    mds_dir=_doc_svc._mds_dir,
                    pdf_path=pdf_path,
                    service=svc,
                    on_progress=_on_progress,
                    stop_check=_stop_check,
                    concurrency=concurrency,
                    force_regenerate=body.force,
                )

                # Re-load so we serialise the same record the pipeline
                # would see — the builder's return value is just the
                # ``DocumentSummary``, but the response needs the full
                # metadata (user_edited, generated_at, …).
                stored = store.load()
                if stored is None:
                    raise RuntimeError("Summary was generated but not persisted")
                payload = _stored_to_response(md_name, doc_stem, stored).model_dump()
                await queue.put({"type": "done", "summary": payload})
            except (asyncio.CancelledError, InterruptedError):
                await queue.put({"type": "cancelled"})
            except HTTPException as exc:
                await queue.put({"type": "error", "status": exc.status_code, "message": exc.detail})
            except Exception:
                logger.exception("Unexpected error during summary generation")
                await queue.put({"type": "error", "status": 500,
                                 "message": "An internal error occurred. Check server logs."})
            finally:
                await queue.put(None)

        runner = asyncio.create_task(run_summary())

        async def _cancel() -> None:
            cancelled_flag["v"] = True
            if not runner.done():
                runner.cancel()
                await asyncio.gather(runner, return_exceptions=True)

        def _handle(event: dict) -> tuple[str, bool]:
            etype = event.get("type")
            if etype == "error":
                return _sse_error(event["status"], event["message"]), True
            if etype == "cancelled":
                return _sse({"type": "cancelled"}), True
            return _sse(event), etype == "done"

        async for frame in run_sse_event_loop(
            queue=queue,
            http_request=http_request,
            on_cancel=_cancel,
            handle_event=_handle,
            watchdog_s=watchdog_s,
            queue_timeout_s=queue_timeout_s,
            log_name="summary generation",
        ):
            yield frame

    return StreamingResponse(event_stream(), media_type="text/event-stream")
