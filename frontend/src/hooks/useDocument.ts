import { useState, useEffect, useCallback, useRef } from 'react'
import type {
  DocumentData,
  ConverterType,
  VLMSettings,
  CloudSettings,
  MarkdownVersion,
  ChunksVersion,
} from '../types'
import { parseSse, CONNECTION_LOST_MSG } from '../utils/parseSse'
import { API_BASE } from '../services/apiService'
import { listMarkdownVersions, getMarkdownContent } from '../services/markdownsApi'
import { listChunksVersions } from '../services/chunksApi'

export type BulkProgressFn = (current: number, total: number, filename: string) => void
export type BulkResultFn = (filename: string, success: boolean) => void

const API = API_BASE

export interface ToastCallbacks {
  onSuccess: (msg: string) => void
  onError: (msg: string) => void
}

export interface ConversionProgress {
  active: boolean
  current: number
  total: number
}

// ─────────────────────────────────────────────────────────────
// useDocument
// ─────────────────────────────────────────────────────────────
export function useDocument(toast: ToastCallbacks) {
  const [documents, setDocuments] = useState<string[]>([])
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null)
  const [documentData, setDocumentData] = useState<DocumentData | null>(null)
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [converting, setConverting] = useState(false)
  const [convertingToPdf, setConvertingToPdf] = useState(false)
  const [savingMd, setSavingMd] = useState(false)
  const [conversionProgress, setConversionProgress] = useState<ConversionProgress | null>(null)
  const [conversionErrorMessage, setConversionErrorMessage] = useState<string | null>(null)

  // ── Versioned markdowns / chunks ────────────────────────────────────────────
  // Lists of every saved Markdown / chunks version for the active document.
  // The "selected" identifiers track which version is currently displayed in
  // the viewer; defaults to the first available entry on document switch.
  const [availableMarkdowns, setAvailableMarkdowns] = useState<MarkdownVersion[]>([])
  const [availableChunks, setAvailableChunks] = useState<ChunksVersion[]>([])
  const [selectedMarkdown, setSelectedMarkdown] = useState<string | null>(null)
  const [selectedChunks, setSelectedChunks] = useState<string | null>(null)

  // Keep latest toast in a ref so all stable useCallback closures always
  // call the current toast functions without needing them as dependencies.
  const toastRef = useRef<ToastCallbacks>(toast)
  toastRef.current = toast

  // Track the currently selected filename and document data in refs so callbacks
  // always see the latest values without needing them in their dependency arrays.
  const selectedDocRef = useRef<string | null>(null)
  selectedDocRef.current = selectedDoc
  const documentDataRef = useRef<DocumentData | null>(null)
  documentDataRef.current = documentData

  // AbortControllers for in-flight requests
  const fetchDocAbortRef = useRef<AbortController | null>(null)
  const convertAbortRef = useRef<AbortController | null>(null)
  const convertToPdfAbortRef = useRef<AbortController | null>(null)

  const fetchDocuments = useCallback(async () => {
    try {
      const res = await fetch(`${API}/documents`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: string[] = await res.json()
      setDocuments(data)
    } catch {
      toastRef.current.onError('Failed to fetch document list')
    }
  }, [])

  useEffect(() => { fetchDocuments() }, [fetchDocuments])

  /**
   * Fetch the markdowns + chunks version lists for the active document.
   * Returns the markdowns array so callers can sync ``selectedMarkdown``
   * with the resolved default (matching the md_filename returned by
   * ``GET /document/{filename}``).
   */
  const fetchVersions = useCallback(async (filename: string): Promise<MarkdownVersion[]> => {
    const [mdVersions, chunkVersions] = await Promise.all([
      listMarkdownVersions(filename).catch(() => []),
      listChunksVersions(filename).catch(() => []),
    ])
    setAvailableMarkdowns(mdVersions)
    setAvailableChunks(chunkVersions)
    return mdVersions
  }, [])

  /** Replace the currently displayed Markdown with another available version. */
  const selectMarkdownVersion = useCallback(async (identifier: string) => {
    const filename = selectedDocRef.current
    if (!filename) return
    try {
      const data = await getMarkdownContent(filename, identifier)
      setSelectedMarkdown(data.filename)
      setDocumentData(prev => prev ? { ...prev, md_filename: data.filename, md_content: data.content, has_markdown: true } : prev)
    } catch {
      toastRef.current.onError('Failed to load Markdown version')
    }
  }, [])

  /** Re-fetch document data for the currently selected document (e.g. after batch conversion). */
  const refreshDocument = useCallback(async () => {
    const filename = selectedDocRef.current
    if (!filename) return

    fetchDocAbortRef.current?.abort()
    fetchDocAbortRef.current = new AbortController()

    setLoading(true)
    try {
      const res = await fetch(
        `${API}/document/${encodeURIComponent(filename)}`,
        { signal: fetchDocAbortRef.current.signal },
      )
      if (!res.ok) throw new Error()
      const data: DocumentData = await res.json()
      setDocumentData(data)
      fetchVersions(filename).then(versions => {
        const match = versions.find(v => v.filename === data.md_filename)
        setSelectedMarkdown(match?.filename ?? versions[0]?.filename ?? null)
      }).catch(() => {})
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
    } finally {
      setLoading(false)
    }
  }, [fetchVersions])

  const selectDocument = useCallback(async (filename: string) => {
    if (filename === selectedDocRef.current) return

    fetchDocAbortRef.current?.abort()
    fetchDocAbortRef.current = new AbortController()

    convertAbortRef.current?.abort()
    setConverting(false)
    setConversionErrorMessage(null)

    setSelectedDoc(filename)
    setDocumentData(null)
    setAvailableMarkdowns([])
    setAvailableChunks([])
    setSelectedMarkdown(null)
    setSelectedChunks(null)
    setLoading(true)
    try {
      const res = await fetch(
        `${API}/document/${encodeURIComponent(filename)}`,
        { signal: fetchDocAbortRef.current.signal },
      )
      if (!res.ok) throw new Error()
      const data: DocumentData = await res.json()
      setDocumentData(data)
      // Kick off the version-list fetch in parallel; doesn't block first paint.
      fetchVersions(filename).then(versions => {
        const match = versions.find(v => v.filename === data.md_filename)
        setSelectedMarkdown(match?.filename ?? versions[0]?.filename ?? null)
      }).catch(() => {})
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      toastRef.current.onError(`Failed to load "${filename}"`)
    } finally {
      setLoading(false)
    }
  }, [fetchVersions])

  const uploadFiles = useCallback(async (files: File[]) => {
    if (files.length === 0) return
    setUploading(true)
    try {
      const formData = new FormData()
      files.forEach(f => formData.append('files', f))
      const res = await fetch(`${API}/upload`, { method: 'POST', body: formData })
      if (!res.ok) {
        const text = await res.text().catch(() => '')
        let detail = 'Upload failed'
        try { detail = (JSON.parse(text) as { detail?: string })?.detail ?? (text || detail) } catch { detail = text || detail }
        throw new Error(detail)
      }
      await fetchDocuments()
      toastRef.current.onSuccess(`Uploaded ${files.length} file${files.length > 1 ? 's' : ''}`)
    } catch (err) {
      toastRef.current.onError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }, [fetchDocuments])

  const deleteDocuments = useCallback(async (filenames: string[]) => {
    if (filenames.length === 0) return
    try {
      const res = await fetch(`${API}/documents`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filenames),
      })
      if (!res.ok) throw new Error()
      if (selectedDocRef.current && filenames.includes(selectedDocRef.current)) {
        setSelectedDoc(null)
        setDocumentData(null)
      }
      await fetchDocuments()
      toastRef.current.onSuccess(`Deleted ${filenames.length} document${filenames.length > 1 ? 's' : ''}`)
    } catch {
      toastRef.current.onError('Delete failed')
    }
  }, [fetchDocuments])

  const convertToMarkdown = useCallback(async (
    converter: ConverterType = 'pymupdf',
    vlm?: VLMSettings,
    cloud?: CloudSettings,
  ) => {
    if (!selectedDocRef.current) return

    convertAbortRef.current?.abort()
    const abortCtrl = new AbortController()
    convertAbortRef.current = abortCtrl

    setConverting(true)
    setConversionProgress({ active: true, current: 0, total: 0 })
    setConversionErrorMessage(null)
    try {
      const body: Record<string, unknown> = {
        filenames: [selectedDocRef.current],
        converter,
      }
      if (converter === 'vlm' && vlm) body.vlm = vlm
      if (converter === 'cloud' && cloud) body.cloud = cloud

      const res = await fetch(`${API}/convert`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: abortCtrl.signal,
      })
      if (!res.ok) {
        const text = await res.text().catch(() => '')
        throw new Error(`HTTP ${res.status}: ${text}`)
      }
      if (!res.body) throw new Error('No response body')

      let mdContent: string | undefined
      let mdFilename: string | undefined
      let fileError: string | undefined
      for await (const event of parseSse(res.body, () => setConversionErrorMessage(CONNECTION_LOST_MSG))) {
        if (event.type === 'progress') {
          setConversionProgress({ active: true, current: event.current as number, total: event.total as number })
        } else if (event.type === 'file_done') {
          if (!event.success) { fileError = String(event.error ?? 'Conversion failed'); break }
          mdContent = event.md_content as string
          mdFilename = event.md_filename as string | undefined
          break
        } else if (event.type === 'error') {
          throw new Error(String(event.message ?? 'Conversion error'))
        } else if (event.type === 'cancelled') {
          throw new DOMException('Conversion cancelled', 'AbortError')
        }
      }

      if (fileError) throw new Error(fileError)
      if (mdContent === undefined) throw new Error('Stream ended without a result')
      setDocumentData(prev => prev
        ? { ...prev, has_markdown: true, md_content: mdContent!, md_filename: mdFilename ?? prev.md_filename }
        : prev)
      toastRef.current.onSuccess('Conversion complete ✓')
      // A new converter-suffixed file may have been written — refresh the
      // version list AND sync the dropdown selection so it points to the file
      // that's actually being displayed (instead of whichever variant was
      // selected before the conversion).
      const fn = selectedDocRef.current
      if (fn) {
        fetchVersions(fn).then(versions => {
          if (!mdFilename) return
          const match = versions.find(v => v.filename === mdFilename)
          setSelectedMarkdown(match?.filename ?? mdFilename)
        }).catch(() => {})
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      toastRef.current.onError(err instanceof Error ? err.message : 'Conversion failed')
    } finally {
      if (convertAbortRef.current === abortCtrl) {
        setConverting(false)
        setConversionProgress(null)
        setConversionErrorMessage(null)
      }
    }
  }, [fetchVersions])

  const cancelConversion = useCallback(() => {
    convertAbortRef.current?.abort()
    setConverting(false)
    setConversionProgress(null)
    setConversionErrorMessage(null)
  }, [])

  const convertMdToPdf = useCallback(async () => {
    if (!selectedDocRef.current) return

    const docAtStart = selectedDocRef.current
    convertToPdfAbortRef.current?.abort()
    const abortCtrl = new AbortController()
    convertToPdfAbortRef.current = abortCtrl

    setConvertingToPdf(true)
    try {
      const res = await fetch(
        `${API}/md-to-pdf/${encodeURIComponent(docAtStart)}`,
        { method: 'POST', signal: abortCtrl.signal },
      )
      if (!res.ok) throw new Error()
      const data = await res.json()
      if (typeof data.pdf_filename !== 'string') throw new Error('Invalid response: missing pdf_filename')
      if (selectedDocRef.current === docAtStart) setSelectedDoc(data.pdf_filename)
      setDocumentData(prev => prev ? { ...prev, has_pdf: true, pdf_filename: data.pdf_filename } : prev)
      await fetchDocuments()
      toastRef.current.onSuccess('Converted to PDF ✓')
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      toastRef.current.onError('MD to PDF conversion failed')
    } finally {
      if (convertToPdfAbortRef.current === abortCtrl) {
        setConvertingToPdf(false)
      }
    }
  }, [fetchDocuments])

  const cancelMdToPdfConversion = useCallback(() => {
    convertToPdfAbortRef.current?.abort()
    setConvertingToPdf(false)
  }, [])

  const saveMarkdown = useCallback(async (content: string) => {
    if (!selectedDocRef.current) return
    setSavingMd(true)
    try {
      const mdFilename =
        documentDataRef.current?.md_filename ??
        selectedDocRef.current.replace(/\.pdf$/i, '.md')
      const file = new File([new Blob([content], { type: 'text/markdown' })], mdFilename, { type: 'text/markdown' })
      const formData = new FormData()
      formData.append('files', file)
      const res = await fetch(`${API}/upload`, { method: 'POST', body: formData })
      if (!res.ok) throw new Error()
      setDocumentData(prev => prev ? { ...prev, md_content: content } : prev)
      toastRef.current.onSuccess('Markdown saved ✓')
    } catch {
      toastRef.current.onError('Failed to save Markdown')
    } finally {
      setSavingMd(false)
    }
  }, [])

  const deleteMarkdown = useCallback(async () => {
    const filename = selectedDocRef.current
    if (!filename) return
    const mdFilename =
      documentDataRef.current?.md_filename ??
      filename.replace(/\.pdf$/i, '.md')
    try {
      const res = await fetch(`${API}/documents`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify([mdFilename]),
      })
      if (!res.ok) throw new Error()
      // Other converter variants may still be on disk; re-fetch the version
      // list and either fall through to a sibling version or clear the
      // viewer if this was the last one.
      const remaining = await fetchVersions(filename)
      if (remaining.length === 0) {
        setDocumentData(prev => prev ? { ...prev, has_markdown: false, md_content: '' } : prev)
        setSelectedMarkdown(null)
      } else {
        const next = remaining[0]
        const identifier = next.source === 'converted' && next.converter
          ? next.converter
          : next.filename
        await selectMarkdownVersion(identifier)
      }
      toastRef.current.onSuccess('Markdown removed — ready to reconvert')
    } catch {
      toastRef.current.onError('Failed to remove Markdown')
    }
  }, [fetchVersions, selectMarkdownVersion])

  const batchConvert = useCallback(async (
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
  ): Promise<void> => {
    const body: Record<string, unknown> = { filenames, converter }
    if (converter === 'vlm' && vlm) body.vlm = vlm
    if (converter === 'cloud' && cloud) body.cloud = cloud

    const res = await fetch(`${API}/convert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`HTTP ${res.status}: ${text}`)
    }
    if (!res.body) throw new Error('No response body')

    for await (const event of parseSse(res.body, onConnectionLost)) {
      if (event.type === 'progress') {
        onPageProgress?.(
          event.filename as string,
          event.current as number,
          event.total as number,
          event.file_index as number,
          event.file_total as number,
        )
      } else if (event.type === 'file_start') {
        onFileStart(
          event.filename as string,
          event.index as number,
          event.total as number,
        )
      } else if (event.type === 'file_done') {
        onFileResult(event.filename as string, event.success as boolean)
      } else if (event.type === 'file_progress') {
        onBatchProgress(
          event.current as number,
          event.total as number,
          event.filename as string,
          event.percentage as number,
        )
        await new Promise<void>(r => setTimeout(r, 0))
      } else if (event.type === 'batch_done' || event.type === 'cancelled') {
        return
      }
    }
  }, [])

  return {
    documents, selectedDoc, documentData, loading, uploading, converting, convertingToPdf, savingMd,
    conversionProgress, conversionErrorMessage,
    selectDocument, refreshDocument, uploadFiles, deleteDocuments,
    convertToMarkdown, cancelConversion,
    convertMdToPdf, cancelMdToPdfConversion,
    saveMarkdown, deleteMarkdown,
    batchConvert,
    availableMarkdowns, availableChunks,
    selectedMarkdown, selectedChunks,
    selectMarkdownVersion,
    setSelectedChunks,
    refreshVersions: () => {
      const fn = selectedDocRef.current
      if (fn) fetchVersions(fn).catch(() => {})
    },
  }
}
