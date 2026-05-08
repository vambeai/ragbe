import { useState, useRef, useCallback } from 'react'
import type { ChunkSettings, ConverterType, VLMSettings, CloudSettings, Chunk } from '../types'
import type { BulkProgressFn, BulkResultFn } from './useDocument'
import { chunkSse, saveChunks as saveChunksApi } from '../services/chunksApi'
import { CONNECTION_LOST_MSG } from '../utils/parseSse'

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
    onFileResult: (filename: string, success: boolean) => void,
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
        (filename, success) => {
          onResult(filename, success)
          if (success) { succeeded++; succeededFiles.add(filename) }
          else failed++
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
    if (succeeded > 0) showToast(`Converted ${succeeded} file${succeeded > 1 ? 's' : ''} ✓`, 'success')
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

  return { bulkOp, bulkConnectionLost, interruptBulk, handleBulkConvert, handleBulkChunk }
}
