"""
Router for chunking endpoints.

Prefix: /api

POST /api/chunk
    Accepts one or more filenames. Loads each document's saved Markdown from
    disk, splits it using the requested strategy and library, saves the resulting
    chunks, and streams progress via Server-Sent Events.

    Runs in a dedicated ProcessPoolExecutor (cpu_chunker_executor) so that
    CPU-bound splitting work runs in isolated processes — no shared GIL, true
    parallelism when multiple documents are chunked concurrently.

    SSE event types (consistent for 1 or N files):
        {"type": "file_start",  "filename": "...", "index": 1, "total": N}
        {"type": "file_done",   "filename": "...", "success": true,
         "total_chunks": N, "chunker_type": "...", "chunker_library": "...",
         "chunks": [...]}
        {"type": "file_done",   "filename": "...", "success": false, "error": "..."}
        {"type": "file_progress","filename": "...", "current": 1, "total": N, "percentage": 50}
        {"type": "batch_done",  "succeeded": N, "failed": M}
        {"type": "error",       "status": 4xx/5xx, "message": "..."}
        {"type": "cancelled"}

GET  /api/chunks/load/{filename}
POST /api/chunks/save
    Standard JSON endpoints for persisting and retrieving chunk sets.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

from backend.config import get_settings
from backend.models.schemas import (
    ChunkFilesRequest,
    ChunksVersionsResponse,
    LoadChunksResponse,
    SaveChunksRequest,
    SaveChunksResponse,
)
from backend.services.chunk_storage_service import ChunkStorageService
from backend.services.chunking_service import _init_chunk_worker, chunk_file_in_process
from backend.utils.executor import cancel_cpu_executor
from backend.utils.sse import (
    run_sse_event_loop,
    sse_error as _sse_error,
    sse_event as _sse,
)

router = APIRouter(prefix="/api", tags=["chunks"])
_storage = ChunkStorageService()


@router.post("/chunk")
async def chunk_documents(http_request: Request, request: ChunkFilesRequest):
    """Chunk one or more documents, streaming progress via SSE.

    Each document's Markdown is loaded from disk, split with the requested
    strategy and library, saved, and its result streamed as SSE events.
    Jobs run concurrently in a ProcessPoolExecutor (up to MAX_CONCURRENT_CHUNKING).

    SSE events: ``file_start`` → ``file_done`` × N → ``batch_done``
    (or ``error`` / ``cancelled``).
    """

    async def event_stream() -> AsyncGenerator[str, None]:
        filenames = request.filenames
        total = len(filenames)
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        succeeded = 0
        failed = 0
        _lock = asyncio.Lock()

        settings_dict = {
            "chunker_type": request.chunker_type.value,
            "chunker_library": request.chunker_library.value,
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "enable_markdown_sizing": request.enable_markdown_sizing,
            # Worker-only transport key — chunking_service strips it before
            # forwarding to ChunkRequest.  Only meaningful for single-file
            # requests; harmless when None.
            "md_filename": request.md_filename if total == 1 else None,
        }

        _settings = get_settings()
        watchdog_s = _settings.SSE_WATCHDOG_TIMEOUT_S
        queue_timeout_s = _settings.SSE_QUEUE_GET_TIMEOUT_S
        cancel_wait_s = _settings.SSE_CANCEL_WAIT_TIMEOUT_S
        # The executor is now read freshly inside _submit_with_retry below,
        # so a sibling cancellation that retires the pool doesn't strand us
        # holding a reference to the dead one.
        semaphore = http_request.app.state.chunk_semaphore
        _cpu_futures: list = []

        if await http_request.is_disconnected():
            yield _sse({"type": "cancelled"})
            return

        async def _submit_with_retry(fn: str) -> dict:
            """Submit a chunk job and tolerate one BrokenProcessPool.

            When the user switches documents quickly, the disconnect on
            the abandoned request triggers `_cancel_all` which SIGTERMs
            every worker in the pool — including any worker about to pick
            up *this* request.  ``cancel_cpu_executor`` swaps the executor
            on app.state in the same teardown, so a single retry against
            the fresh pool succeeds.
            """
            last_exc: Exception | None = None
            for attempt in range(2):
                cur_executor: ProcessPoolExecutor = http_request.app.state.cpu_chunker_executor
                cf_local = cur_executor.submit(chunk_file_in_process, fn, settings_dict)
                _cpu_futures.append(cf_local)
                try:
                    return await asyncio.wrap_future(cf_local)
                except BrokenProcessPool as exc:
                    last_exc = exc
                    logger.info(
                        "Chunk worker pool was retired mid-request for '%s' (attempt %d) — retrying on fresh pool",
                        fn, attempt + 1,
                    )
                finally:
                    try:
                        _cpu_futures.remove(cf_local)
                    except ValueError:
                        pass
            assert last_exc is not None
            raise last_exc

        async def chunk_one(idx: int, fn: str) -> None:
            nonlocal succeeded, failed

            async with semaphore:
                if await http_request.is_disconnected():
                    return

                queue.put_nowait({"type": "file_start", "filename": fn, "index": idx + 1, "total": total})

                _done = 0
                try:
                    result = await _submit_with_retry(fn)

                    async with _lock:
                        if result.get("success"):
                            succeeded += 1
                        else:
                            failed += 1
                        _done = succeeded + failed
                    # Use put_nowait for both so no coroutine can interleave
                    # between them and produce out-of-order file_progress %.
                    queue.put_nowait({"type": "file_done", "filename": fn, **result})

                except Exception as exc:
                    async with _lock:
                        failed += 1
                        _done = succeeded + failed
                    error_summary = f"{type(exc).__name__}: {str(exc)[:120]}"
                    queue.put_nowait({"type": "file_done", "filename": fn, "success": False, "error": error_summary})
                    logger.warning("Chunk failed for '%s': %s", fn, exc, exc_info=True)

                queue.put_nowait({
                    "type": "file_progress",
                    "filename": fn,
                    "current": _done,
                    "total": total,
                    "percentage": round(_done / total * 100),
                })

        async def run_all() -> None:
            tasks = [asyncio.create_task(chunk_one(i, fn)) for i, fn in enumerate(filenames)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                    logger.error("Unexpected exception from chunk task: %s", res, exc_info=res)
            await queue.put(None)

        runner = asyncio.create_task(run_all())

        async def _cancel_all() -> None:
            s = get_settings()
            await cancel_cpu_executor(
                _cpu_futures,
                http_request.app.state,
                "cpu_chunker_executor",
                s.MAX_CONCURRENT_CHUNKING,
                _init_chunk_worker,
                "chunk worker",
                logger,
            )
            runner.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(runner), timeout=cancel_wait_s)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        async def _safe_cancel() -> None:
            # Idempotent wrapper: skip the (potentially expensive) executor
            # swap when the runner has already finished.  This matches the
            # original "finally only runs if not runner.done()" safety-net
            # semantics now that the helper invokes on_cancel in finally.
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
            log_name=f"chunking ({total} job(s))",
            on_complete=_on_complete,
        ):
            yield frame

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/chunks/save", response_model=SaveChunksResponse)
async def save_chunks(request: SaveChunksRequest):
    """Persist a chunk set to a timestamped JSON file on disk."""
    return await asyncio.to_thread(_storage.save_chunks, request)


@router.get("/chunks/load/{filename}", response_model=LoadChunksResponse)
async def load_chunks(filename: str):
    """Load the most recently saved chunk set for a document."""
    return await asyncio.to_thread(_storage.load_chunks, filename)


@router.get(
    "/documents/{document_name}/chunks",
    response_model=ChunksVersionsResponse,
)
async def list_chunks_versions(document_name: str):
    """Return every saved chunks JSON file for a document, newest first.

    Each entry has its ``algorithm`` and ``timestamp`` parsed from the
    filename so the frontend can render a human-readable picker.  Legacy
    files that pre-date the new naming scheme appear with
    ``algorithm = "unknown"``.
    """
    versions = await asyncio.to_thread(_storage.list_versions, document_name)
    return ChunksVersionsResponse(document_name=document_name, versions=versions)


@router.get(
    "/documents/{document_name}/chunks/{chunks_filename}",
    response_model=LoadChunksResponse,
)
async def load_chunks_version(document_name: str, chunks_filename: str):
    """Load one specific saved-chunks file by filename."""
    return await asyncio.to_thread(
        _storage.load_chunks_by_filename, document_name, chunks_filename,
    )
