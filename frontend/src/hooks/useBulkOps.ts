import { useState, useRef, useCallback } from 'react'
import type { ChunkSettings, ConverterType, VLMSettings, CloudSettings, Chunk } from '../types'
import type { BulkProgressFn, BulkResultFn } from './useDocument'
import { apiEnrichMarkdownPipeline, API_BASE } from '../services/apiService'
import { listMarkdownVersions } from '../services/markdownsApi'
import { chunkSse, saveChunks as saveChunksApi } from '../services/chunksApi'
import { CONNECTION_LOST_MSG } from '../utils/parseSse'

/**
 * Priority order used by bulk enrichment when a PDF has multiple
 * converter variants on disk.  Higher in the list wins.  Variants not
 * in this list are skipped if any listed one is present; if NONE of
 * the listed variants exist we fall through to whichever ``converted``
 * variant the backend returned first (rare — keeps the bulk run from
 * silently dropping a file just because its converter was renamed).
 */
const BULK_ENRICH_CONVERTER_PRIORITY = ['vlm', 'cloud', 'docling', 'markitdown', 'liteparse', 'pymupdf4llm']

export interface BulkOp {
  title: string
  detail: string
  current: number
  total: number
}

interface Options {
  batchConvert: (
    filenames: string[],
    converter: ConverterType,
    vlm: VLMSettings | undefined,
    cloud: CloudSettings | undefined,
    onFileStart: (filename: string, index: number, total: number) => void,
    onFileResult: (filename: string, success: boolean, failedPages?: number[]) => void,
    onBatchProgress: (current: number, total: number, filename: string, percentage: number) => void,
    signal?: AbortSignal,
    onConnectionLost?: () => void,
    onPageProgress?: (filename: string, page: number, totalPages: number, fileIndex: number, fileTotal: number) => void,
  ) => Promise<void>
  settings: ChunkSettings
  showToast: (message: string, type: 'success' | 'error') => void
  onConvertSuccess: (succeededFiles: Set<string>) => Promise<void>
}

export function useBulkOps({
  batchConvert,
  settings,
  showToast,
  onConvertSuccess,
}: Options) {
  const [bulkOp, setBulkOp] = useState<BulkOp | null>(null)
  const [bulkConnectionLost, setBulkConnectionLost] = useState(false)
  const bulkAbortRef = useRef<AbortController | null>(null)

  const interruptBulk = useCallback(() => {
    bulkAbortRef.current?.abort()
    setBulkConnectionLost(false)
  }, [])

  const handleBulkConvert = useCallback(async (
    filenames: string[],
    onProgress: BulkProgressFn,
    onResult: BulkResultFn,
  ) => {
    bulkAbortRef.current?.abort()
    bulkAbortRef.current = new AbortController()

    setBulkOp({ title: 'Batch PDF → Markdown', detail: '', current: 0, total: filenames.length })
    setBulkConnectionLost(false)

    let succeeded = 0
    let failed = 0
    // ``partial`` counts files whose conversion finished (``success: true``)
    // but left at least one failed page behind — see the VLM converter's
    // graceful-failure logic.  We track it separately so the completion
    // toast distinguishes "10 clean" from "7 clean + 3 with partial failures",
    // which is otherwise invisible until the user opens each file.
    let partial = 0
    const succeededFiles = new Set<string>()

    try {
      await batchConvert(
        filenames,
        settings.converter as ConverterType,
        settings.vlm as VLMSettings | undefined,
        settings.cloud,
        (filename, index, total) => {
          setBulkOp(prev => prev
            ? { ...prev, detail: `File ${index} of ${total} — ${filename}` }
            : null
          )
        },
        (filename, success, failedPages) => {
          onResult(filename, success, failedPages)
          if (success) {
            succeeded++
            succeededFiles.add(filename)
            if (failedPages && failedPages.length > 0) partial++
          } else {
            failed++
          }
        },
        (current, total, filename, _percentage) => {
          onProgress(current, total, filename)
          setBulkOp(prev => prev ? { ...prev, current } : null)
        },
        bulkAbortRef.current.signal,
        () => setBulkConnectionLost(true),
        (filename, page, totalPages, fileIndex, fileTotal) => {
          setBulkOp(prev => prev
            ? {
                ...prev,
                detail: `Converting page ${page} of ${totalPages} — ${filename} (file ${fileIndex} of ${fileTotal})`,
              }
            : null
          )
        },
      )
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        showToast(err instanceof Error ? err.message : 'Batch conversion failed', 'error')
        setBulkOp(null)
        setBulkConnectionLost(false)
        return
      }
    }

    setBulkOp(null)
    setBulkConnectionLost(false)
    if (succeeded > 0) {
      // Two toasts when the batch is mixed: a clean success count, plus a
      // separate warning so the partial files don't get hidden behind a
      // single "10/10 ✓" message.
      const cleanCount = succeeded - partial
      if (cleanCount > 0) showToast(`Converted ${cleanCount} file${cleanCount > 1 ? 's' : ''} ✓`, 'success')
      if (partial > 0) showToast(
        `${partial} file${partial > 1 ? 's' : ''} converted with failed pages — see ⚠ in the sidebar`,
        'error',
      )
    }
    if (failed > 0) showToast(`${failed} file${failed > 1 ? 's' : ''} failed to convert`, 'error')

    if (succeeded > 0) await onConvertSuccess(succeededFiles)
  }, [batchConvert, settings, showToast, onConvertSuccess])

  const handleBulkChunk = useCallback(async (
    filenames: string[],
    onProgress: BulkProgressFn,
    onResult: BulkResultFn,
  ) => {
    bulkAbortRef.current?.abort()
    bulkAbortRef.current = new AbortController()
    const { signal } = bulkAbortRef.current

    setBulkOp({ title: 'Batch Chunking', detail: '', current: 0, total: filenames.length })

    let succeeded = 0
    let failed = 0
    let saveFailed = 0

    const onFileStart = (filename: string, index: number, total: number) => {
      onProgress(index, total, filename)
      setBulkOp(prev => prev
        ? { ...prev, detail: `File ${index} of ${total} — ${filename}`, current: index }
        : null
      )
    }

    // The backend no longer auto-saves chunks during /api/chunk; save must be
    // triggered explicitly by the caller for each successfully chunked file.
    // Bulk chunking from the sidebar IS the explicit "bulk export" pathway,
    // so we save every successful file as it completes.  The backend echoes
    // back the md_filename it actually chunked, so we forward it to keep
    // chunks generated from different MD variants distinguishable on disk.
    //
    // A save failure must NOT abort the rest of the batch, but it also must
    // not be silently swallowed: counted separately so the user sees a clear
    // "chunked but not saved" toast at the end (otherwise the success toast
    // misleads them into thinking every file made it to disk).
    const onFileDone = async (filename: string, success: boolean, chunks: Chunk[], mdFilename: string | null) => {
      onResult(filename, success)
      if (success) {
        succeeded++
        try {
          await saveChunksApi({ filename, mdFilename, settings, chunks })
        } catch (err) {
          saveFailed++
          console.warn(`Failed to save chunks for '${filename}':`, err)
        }
      } else {
        failed++
      }
    }

    try {
      await chunkSse(
        filenames,
        settings,
        signal,
        () => setBulkConnectionLost(true),
        onFileStart,
        onFileDone,
      )
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        showToast(err instanceof Error ? err.message : 'Batch chunking failed', 'error')
      }
    }

    setBulkOp(null)
    setBulkConnectionLost(false)
    const persisted = succeeded - saveFailed
    if (persisted > 0) showToast(`Chunked ${persisted} file${persisted > 1 ? 's' : ''} ✓`, 'success')
    if (failed > 0) showToast(`${failed} file${failed > 1 ? 's' : ''} failed to chunk`, 'error')
    if (saveFailed > 0) showToast(`${saveFailed} file${saveFailed > 1 ? 's' : ''} chunked but not saved`, 'error')
  }, [settings, showToast])

  // ── Bulk markdown enrichment (Phase C) ───────────────────────────────────
  //
  // Iterates the selected PDF filenames one at a time, picks the
  // highest-priority markdown variant present for each, runs the
  // enrichment pipeline against it, and persists the corrected output.
  // Each pipeline call uses whatever document summary is already cached
  // on disk — bulk DOES NOT open the SummaryReviewModal (the user
  // confirmed "bulk skip" — the curated review path is reserved for
  // single-doc enrichment).  Files with no markdown are skipped with a
  // failure result; per-file failures never abort the rest of the run.
  //
  // We always pass ``use_summary=true`` so the backend silently
  // attaches a cached summary when one exists.  The per-doc
  // ``skip_summary`` setting only affects the single-doc modal flow —
  // honoring it here would couple bulk to UI state that isn't visible
  // in the bulk context.
  const handleBulkEnrich = useCallback(async (
    filenames: string[],
    onProgress: BulkProgressFn,
    onResult: BulkResultFn,
  ) => {
    const enrichSettings = settings.sectionEnrichment
    if (!enrichSettings?.model) {
      showToast('Section Enrichment model is not configured', 'error')
      filenames.forEach(f => onResult(f, false))
      return
    }

    bulkAbortRef.current?.abort()
    bulkAbortRef.current = new AbortController()
    const { signal } = bulkAbortRef.current

    setBulkOp({ title: 'Batch Markdown Enrichment', detail: '', current: 0, total: filenames.length })
    setBulkConnectionLost(false)

    let succeeded = 0
    let failed = 0
    let skippedNoMd = 0
    let saveFailed = 0

    for (let i = 0; i < filenames.length; i++) {
      if (signal.aborted) break
      const filename = filenames[i]
      const fileIndex = i + 1
      onProgress(fileIndex, filenames.length, filename)
      setBulkOp(prev => prev
        ? { ...prev, current: fileIndex, detail: `File ${fileIndex} of ${filenames.length} — ${filename}` }
        : null
      )

      // Variant resolution — list versions, then pick by priority.
      let mdFilename: string | null = null
      try {
        const versions = await listMarkdownVersions(filename)
        const converted = versions.filter(v => v.source === 'converted' && v.converter)
        for (const token of BULK_ENRICH_CONVERTER_PRIORITY) {
          const hit = converted.find(v => v.converter === token)
          if (hit) { mdFilename = hit.filename; break }
        }
        if (!mdFilename && converted.length > 0) mdFilename = converted[0].filename
        if (!mdFilename) {
          // Bare-uploaded MDs (source: 'uploaded') are also enrichable.
          const uploaded = versions.find(v => v.source === 'uploaded')
          if (uploaded) mdFilename = uploaded.filename
        }
      } catch (err) {
        console.warn(`Bulk enrich: variant resolution failed for '${filename}':`, err)
      }

      if (!mdFilename) {
        skippedNoMd++
        onResult(filename, false)
        continue
      }

      try {
        const result = await apiEnrichMarkdownPipeline(
          mdFilename,
          enrichSettings,
          true,   // useCheckpoint — reuse per-piece cache where possible
          true,   // useSummary — silently attach cached summary if present
          (progress) => {
            const detail = progress.totalPieces === 0
              ? `File ${fileIndex} of ${filenames.length} — ${filename} (cleaning…)`
              : `File ${fileIndex} of ${filenames.length} — ${filename} (${progress.completedPieces}/${progress.totalPieces} pieces)`
            setBulkOp(prev => prev ? { ...prev, detail } : null)
          },
          signal,
          () => setBulkConnectionLost(true),
        )

        // Auto-accept: persist the corrected content under the same
        // markdown filename.  Mirrors the existing single-doc flow's
        // "Accept" path; in bulk we have no diff modal to ask.
        try {
          const file = new File(
            [new Blob([result.enrichedContent], { type: 'text/markdown' })],
            mdFilename,
            { type: 'text/markdown' },
          )
          const formData = new FormData()
          formData.append('files', file)
          const saveRes = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData, signal })
          if (!saveRes.ok) throw new Error(`HTTP ${saveRes.status}`)
          succeeded++
          onResult(filename, true)
        } catch (saveErr) {
          if (saveErr instanceof DOMException && saveErr.name === 'AbortError') break
          saveFailed++
          console.warn(`Bulk enrich: save failed for '${mdFilename}':`, saveErr)
          onResult(filename, false)
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') break
        failed++
        console.warn(`Bulk enrich: pipeline failed for '${mdFilename}':`, err)
        onResult(filename, false)
      }
    }

    setBulkOp(null)
    setBulkConnectionLost(false)
    if (succeeded > 0) showToast(`Enriched ${succeeded} file${succeeded > 1 ? 's' : ''} ✓`, 'success')
    if (failed > 0) showToast(`${failed} file${failed > 1 ? 's' : ''} failed to enrich`, 'error')
    if (saveFailed > 0) showToast(`${saveFailed} file${saveFailed > 1 ? 's' : ''} enriched but not saved`, 'error')
    if (skippedNoMd > 0) showToast(`${skippedNoMd} file${skippedNoMd > 1 ? 's' : ''} skipped — no markdown found`, 'error')
  }, [settings.sectionEnrichment, showToast])

  return { bulkOp, bulkConnectionLost, interruptBulk, handleBulkConvert, handleBulkChunk, handleBulkEnrich }
}
