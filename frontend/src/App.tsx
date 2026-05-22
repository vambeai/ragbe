import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import Sidebar from './components/layout/Sidebar'
import MarkdownViewer from './components/viewer/MarkdownViewer'
import ChunkViewer from './components/viewer/ChunkViewer'
import SettingsModal from './components/modals/SettingsModal'
import ProgressModal from './components/modals/ProgressModal'
import ConfirmDialog from './components/modals/ConfirmDialog'
import Toast from './components/Toast'
import { ErrorBoundary } from './components/ErrorBoundary'
import { useDocument } from './hooks/useDocument'
import { useChunks } from './hooks/useChunks'
import { useBulkOps } from './hooks/useBulkOps'
import { loadSplitPct, saveSplitPct } from './hooks/useSettings'
import PDFViewer from './components/viewer/PDFViewer'
import {
  converterFilenameToken,
  fetchDocumentMetadata,
  mdSourceFromFilename,
} from './services/apiService'
import {
  TOAST_DURATION_ERROR_MS,
  TOAST_DURATION_SUCCESS_MS,
} from './config'
import './App.css'

interface ToastState {
  message: string
  type: 'success' | 'error'
  id: number
}

export default function App() {
  // ── Toast ───────────────────────────────────────────────────
  const [toast, setToast] = useState<ToastState | null>(null)

  const toastIdRef = useRef(0)
  const showToast = useCallback((message: string, type: 'success' | 'error') => {
    setToast({ message, type, id: ++toastIdRef.current })
  }, [])

  // Stable reference — prevents Toast's useEffect timer from resetting on re-renders.
  const handleToastClose = useCallback(() => setToast(null), [])

  const toastCallbacks = useMemo(() => ({
    onSuccess: (msg: string) => showToast(msg, 'success'),
    onError: (msg: string) => showToast(msg, 'error'),
  }), [showToast])

  // ── Hooks ────────────────────────────────────────────────────
  const {
    documents, selectedDoc, documentData, loading, uploading, converting, convertingToPdf, savingMd,
    conversionProgress, conversionErrorMessage,
    selectDocument, refreshDocument, uploadFiles, deleteDocuments,
    convertToMarkdown, cancelConversion,
    convertMdToPdf, cancelMdToPdfConversion,
    saveMarkdown, deleteMarkdown,
    batchConvert,
    availableMarkdowns, availableChunks,
    selectedMarkdown, selectedChunks,
    selectMarkdownVersion, setSelectedChunks, refreshVersions,
  } = useDocument(toastCallbacks)

  // ── UI state ─────────────────────────────────────────────────
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [scrollSync, setScrollSync] = useState(true)
  const [leftView, setLeftView] = useState<'pdf' | 'markdown'>('pdf')
  const [rightView, setRightView] = useState<'markdown' | 'chunks'>('markdown')

  // Set of PDF filenames that have a corresponding markdown file.
  const [docsWithMarkdown, setDocsWithMarkdown] = useState<Set<string>>(new Set())

  // Set of PDF filenames whose last VLM run ended with at least one failed
  // page (or was cancelled mid-flight). Source of truth on the backend is
  // the presence of ``.checkpoints/{stem}_vlm/``. Used by the sidebar to
  // show a warning icon so partial conversions are visible at a glance.
  const [docsWithFailures, setDocsWithFailures] = useState<Set<string>>(new Set())

  // The active Markdown source token ("pymupdf4llm", "docling", "uploaded",
  // …) — used by the saved-chunks filter, the auto-link logic inside
  // useChunks, and the chunks-dropdown rendering.
  //
  // Prefers the entry from ``availableMarkdowns`` (authoritative on
  // source/converter) but falls back to parsing the filename so the source
  // is known the moment documentData.md_filename updates — the fetch for
  // the version list is a separate round-trip and would otherwise leave a
  // window where currentMdSource is null and the auto-link can't fire.
  const currentMdSource = useMemo<string | null>(() => {
    if (!documentData?.md_filename) return null
    const v = availableMarkdowns.find(m => m.filename === documentData.md_filename)
    if (v) return v.source === 'converted' && v.converter ? v.converter : 'uploaded'
    return mdSourceFromFilename(documentData.md_filename, documentData.pdf_filename)
  }, [documentData?.md_filename, documentData?.pdf_filename, availableMarkdowns])

  // useChunks derives the active Markdown filename from documentData itself —
  // see the comment in the hook.  We pass it the chunks-version list and
  // the active MD source so it can auto-link the panel to a saved version
  // that already matches the current configuration.
  const {
    chunks, settings, saving: savingChunks, chunking, chunksDirty,
    applySettings, editChunk, deleteChunk, deleteChunks, mergeChunks, saveChunks, cancelChunking,
    enrichChunk, loadSavedChunks, rechunk,
  } = useChunks(
    documentData, selectedDoc, rightView === 'chunks', toastCallbacks,
    availableChunks, currentMdSource, setSelectedChunks,
  )

  const handleSaveChunks = useCallback(async () => {
    const savedFilename = await saveChunks()
    refreshVersions()
    // Highlight the file we just wrote in the saved-versions dropdown — the
    // user expects "I saved this" to translate to "the picker now shows it".
    if (savedFilename) setSelectedChunks(savedFilename)
  }, [saveChunks, refreshVersions, setSelectedChunks])

  // ── Unsaved-chunks confirmation flow ────────────────────────────────────────
  // Any action that would discard the in-memory chunk list (switching docs,
  // changing chunker settings, loading a different saved version) is funneled
  // through ``runOrConfirm`` — when ``chunksDirty`` is true, the action is
  // staged in ``pendingChunkAction`` and the user is asked to Save / Discard /
  // Cancel before it runs.
  type PendingAction =
    | { kind: 'switch'; filename: string }
    | { kind: 'apply-settings'; settings: typeof settings }
    | { kind: 'load-saved'; filename: string }
    | { kind: 'rechunk' }
    | { kind: 'convert' }
    | { kind: 'select-md'; identifier: string }
    | { kind: 'delete-md' }
  const [pendingChunkAction, setPendingChunkAction] = useState<PendingAction | null>(null)

  const executePendingAction = useCallback((a: PendingAction) => {
    if (a.kind === 'switch') {
      selectDocument(a.filename)
    } else if (a.kind === 'apply-settings') {
      applySettings(a.settings)
      // Settings change always triggers a fresh re-chunk in useChunks; clear
      // the saved-version selection so the picker stops claiming we're still
      // viewing the previously-loaded version.
      setSelectedChunks(null)
    } else if (a.kind === 'load-saved') {
      setSelectedChunks(a.filename)
      // Sync the chunker settings to the version being loaded so the next
      // Save overwrites THIS file.  Saved-chunk filenames are
      // configuration-keyed (library/algorithm/size/overlap) — without
      // this sync, edit + save lands at whatever the current settings
      // dictate, NOT at the file the user is actually viewing.
      const v = availableChunks.find(c => c.filename === a.filename)
      if (v) {
        applySettings({
          ...settings,
          chunkerType: v.algorithm,
          chunkerLibrary: v.library,
          chunkSize: v.chunk_size ?? settings.chunkSize,
          chunkOverlap: v.chunk_overlap ?? settings.chunkOverlap,
          // Saved markdown-strategy files only carry size/overlap when
          // markdown sizing was enabled — recover that flag here.
          enableMarkdownSizing: v.algorithm === 'markdown'
            ? v.chunk_size != null
            : settings.enableMarkdownSizing,
        })
      }
      loadSavedChunks(a.filename)
    } else if (a.kind === 'rechunk') {
      rechunk()
      setSelectedChunks(null)
    } else if (a.kind === 'convert') {
      convertToMarkdown(settings.converter, settings.vlm, settings.cloud)
    } else if (a.kind === 'select-md') {
      selectMarkdownVersion(a.identifier)
    } else if (a.kind === 'delete-md') {
      deleteMarkdown()
    }
  }, [selectDocument, applySettings, loadSavedChunks, rechunk, setSelectedChunks,
      convertToMarkdown, selectMarkdownVersion, deleteMarkdown,
      availableChunks, settings])

  const runOrConfirm = useCallback((a: PendingAction) => {
    if (chunksDirty) setPendingChunkAction(a)
    else executePendingAction(a)
  }, [chunksDirty, executePendingAction])

  const handleSaveDirty = useCallback(async () => {
    const a = pendingChunkAction
    if (!a) return
    setPendingChunkAction(null)
    await handleSaveChunks()
    executePendingAction(a)
  }, [pendingChunkAction, handleSaveChunks, executePendingAction])

  const handleDiscardDirty = useCallback(() => {
    const a = pendingChunkAction
    if (!a) return
    setPendingChunkAction(null)
    executePendingAction(a)
  }, [pendingChunkAction, executePendingAction])

  const handleCancelDirty = useCallback(() => setPendingChunkAction(null), [])

  const handleLoadSavedChunks = useCallback((filename: string) => {
    runOrConfirm({ kind: 'load-saved', filename })
  }, [runOrConfirm])

  const handleApplySettings = useCallback((newSettings: typeof settings) => {
    runOrConfirm({ kind: 'apply-settings', settings: newSettings })
  }, [runOrConfirm])

  const handleRechunk = useCallback(() => {
    runOrConfirm({ kind: 'rechunk' })
  }, [runOrConfirm])

  const [settingsOpen, setSettingsOpen] = useState(false)
  const [pdfScale, setPdfScale] = useState(1.0)
  const [mdScale, setMdScale] = useState(1.0)
  const [mdPadding, setMdPadding] = useState(20)
  // Tracks which panel ('left' | 'right') currently has the view-options
  // popover open; null means closed.  Only the MD viewer surfaces this
  // popover (PDF only needs zoom, which sits inline in the panel header).
  const [optionsOpenIn, setOptionsOpenIn] = useState<null | 'left' | 'right'>(null)
  const [splitPct, setSplitPct] = useState(() => loadSplitPct())
  const [isDragging, setIsDragging] = useState(false)
  const leftPanelRef = useRef<HTMLDivElement>(null)
  const rightPanelRef = useRef<HTMLDivElement>(null)
  /** Pending document switch while a conversion/chunking is in progress. */
  const [pendingDoc, setPendingDoc] = useState<string | null>(null)

  // Reset to default layout when the selected doc has no markdown.
  useEffect(() => {
    if (!documentData) return
    if (!documentData.has_markdown) { setLeftView('pdf'); setRightView('markdown') }
  }, [selectedDoc, documentData?.has_markdown])

  // NOTE: a previous version of this file had a useEffect here that cleared
  // ``selectedChunks`` whenever ``selectedMarkdown`` changed.  That ran AFTER
  // useChunks's chunking effect (registration order), so on a doc switch it
  // overwrote the saved-version selection that the auto-link had just set —
  // the dropdown ended up showing the placeholder even when a matching
  // saved file existed.  ``useChunks`` already calls
  // ``setSelectedChunksFilename(null)`` itself when no match is found, so
  // this effect was redundant on top of being broken.

  const handleSetLeftView = (view: 'pdf' | 'markdown') => {
    if (view === 'markdown' && rightView === 'markdown') setRightView('chunks')
    setLeftView(view)
  }

  const handleSetRightView = (view: 'markdown' | 'chunks') => {
    if (view === 'markdown' && leftView === 'markdown') setLeftView('pdf')
    setRightView(view)
  }

  // ── Track which docs have markdown ────────────────────────────
  // Derive a stable string key so the fetch only fires when the document list
  // content changes, not when useDocument recreates the array with the same values.
  const documentsKey = useMemo(() => documents.join(','), [documents])
  useEffect(() => {
    if (documentsKey === '') {
      setDocsWithMarkdown(new Set())
      setDocsWithFailures(new Set())
      return
    }
    fetchDocumentMetadata('fetch document metadata').then(meta => {
      setDocsWithMarkdown(new Set(meta.filter(m => m.has_markdown).map(m => m.filename)))
      setDocsWithFailures(new Set(meta.filter(m => m.has_failures).map(m => m.filename)))
    })
  }, [documentsKey])

  // Keep docsWithMarkdown in sync when a single-file conversion or deletion occurs.
  // Guard against documentData === null (loading state): we don't know yet whether
  // the doc has markdown, so don't prematurely remove it from the set.
  //
  // Intentionally NOT touching ``docsWithFailures`` here: that flag is a
  // per-PDF property (does any variant or checkpoint on disk show failures?)
  // and inferring it from ``documentData.md_content`` would flip the warning
  // off whenever the user switched the version dropdown to a clean variant
  // of a document that still has an unfinished VLM variant on disk.  The
  // ``converting`` effect below refreshes the backend signal at the only
  // moments when ``has_failures`` can actually change for the active doc.
  useEffect(() => {
    if (!selectedDoc || documentData === null) return
    setDocsWithMarkdown(prev => {
      const next = new Set(prev)
      if (documentData.has_markdown) next.add(selectedDoc)
      else next.delete(selectedDoc)
      return next
    })
  }, [selectedDoc, documentData])

  // Refresh metadata (notably ``has_failures``) whenever a single-file
  // conversion finishes.  ``converting`` transitioning true → false is the
  // only point in a single-doc flow at which the on-disk failure state can
  // change.  Bulk convert refreshes via ``handleConvertSuccess``; uploads
  // and deletes refresh via the documents-list effect above; variant
  // switches don't touch on-disk state at all and so deliberately don't
  // trigger this.
  const prevConvertingRef = useRef(converting)
  useEffect(() => {
    if (prevConvertingRef.current && !converting) {
      fetchDocumentMetadata('refresh document metadata after conversion').then(meta => {
        setDocsWithMarkdown(new Set(meta.filter(m => m.has_markdown).map(m => m.filename)))
        setDocsWithFailures(new Set(meta.filter(m => m.has_failures).map(m => m.filename)))
      })
    }
    prevConvertingRef.current = converting
  }, [converting])

  // Re-uploading an existing file leaves the ``documents`` list unchanged,
  // so the ``documentsKey``-driven metadata fetch above doesn't fire. Hook
  // into the uploading false-edge instead so badges (and the selected doc,
  // if it was overwritten) refresh after every upload.
  const prevUploadingRef = useRef(uploading)
  useEffect(() => {
    if (prevUploadingRef.current && !uploading) {
      fetchDocumentMetadata('refresh document metadata after upload').then(meta => {
        setDocsWithMarkdown(new Set(meta.filter(m => m.has_markdown).map(m => m.filename)))
        setDocsWithFailures(new Set(meta.filter(m => m.has_failures).map(m => m.filename)))
      })
      if (selectedDoc) refreshDocument()
    }
    prevUploadingRef.current = uploading
  }, [uploading, selectedDoc, refreshDocument])

  // ── Persist split ratio ───────────────────────────────────────
  const splitPctRef = useRef(splitPct)
  splitPctRef.current = splitPct
  useEffect(() => {
    if (!isDragging) saveSplitPct(splitPctRef.current)
  }, [isDragging])

  const handleToggleScrollSync = useCallback(() => setScrollSync(v => !v), [])

  // ── Divider drag ─────────────────────────────────────────────
  useEffect(() => {
    if (!isDragging) return
    const onMove = (e: MouseEvent) => {
      const container = document.querySelector('.viewers') as HTMLElement
      if (!container) return
      const rect = container.getBoundingClientRect()
      const pct = Math.min(100, Math.max(0, ((e.clientX - rect.left) / rect.width) * 100))
      splitPctRef.current = pct
      if (leftPanelRef.current) leftPanelRef.current.style.width = `${pct}%`
    }
    const onUp = () => {
      setIsDragging(false)
      setSplitPct(splitPctRef.current)
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
    return () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
    }
  }, [isDragging])

  // Funnel every path that could replace or discard the in-memory chunk
  // list through ``runOrConfirm`` so the unsaved-chunks popup fires
  // consistently whether the trigger is a settings change, a doc switch,
  // a Markdown re-conversion, an MD-variant switch, or a delete.
  const handleConvert = useCallback(() => {
    runOrConfirm({ kind: 'convert' })
  }, [runOrConfirm])

  const handleSelectMarkdownVersion = useCallback((identifier: string) => {
    runOrConfirm({ kind: 'select-md', identifier })
  }, [runOrConfirm])

  const handleDeleteMarkdown = useCallback(() => {
    runOrConfirm({ kind: 'delete-md' })
  }, [runOrConfirm])

  const converterLabel = settings.converter ?? 'Convert'

  const conversionDetail = selectedDoc
    ? conversionProgress && conversionProgress.total > 0
      ? `Converting page ${conversionProgress.current} of ${conversionProgress.total} — ${selectedDoc}`
      : selectedDoc
    : ''

  // ── Document selection with in-progress guard ─────────────────
  const handleSelectDocument = useCallback((filename: string) => {
    if (converting || convertingToPdf || chunking) {
      setPendingDoc(filename)
    } else {
      runOrConfirm({ kind: 'switch', filename })
    }
  }, [converting, convertingToPdf, chunking, runOrConfirm])

  const confirmSwitch = useCallback(() => {
    if (!pendingDoc) return
    cancelConversion()
    cancelMdToPdfConversion()
    cancelChunking()
    selectDocument(pendingDoc)
    setPendingDoc(null)
  }, [pendingDoc, cancelConversion, cancelMdToPdfConversion, cancelChunking, selectDocument])

  const cancelSwitch = useCallback(() => setPendingDoc(null), [])

  // ── Bulk operations ───────────────────────────────────────────
  // Called by useBulkOps after a successful batch convert to refresh metadata
  // and the active document panel.
  //
  // After refreshing, we explicitly switch the active document to the variant
  // produced by the converter that was just run.  Without this step
  // refreshDocument falls back to /document/{filename}'s default — the
  // *alphabetically first* converted MD on disk — which means converting a
  // PDF with method B while method A's MD already exists silently leaves the
  // viewer (and the chunking that's keyed off documentData.md_filename) on
  // A's content.
  const handleConvertSuccess = useCallback(async (succeededFiles: Set<string>) => {
    const meta = await fetchDocumentMetadata('refresh document metadata after bulk convert')
    setDocsWithMarkdown(new Set(meta.filter(m => m.has_markdown).map(m => m.filename)))
    setDocsWithFailures(new Set(meta.filter(m => m.has_failures).map(m => m.filename)))
    if (selectedDoc && succeededFiles.has(selectedDoc)) {
      await refreshDocument()
      const token = converterFilenameToken(settings.converter)
      if (token) {
        await selectMarkdownVersion(token)
      }
    }
  }, [selectedDoc, refreshDocument, selectMarkdownVersion, settings.converter])

  const {
    bulkOp, bulkConnectionLost, interruptBulk, handleBulkConvert, handleBulkChunk, handleBulkEnrich,
  } = useBulkOps({
    batchConvert,
    settings,
    showToast,
    onConvertSuccess: handleConvertSuccess,
  })

  // ── Saved-chunk filtering ─────────────────────────────────────────────────
  // The chunks dropdown must reflect ONLY the saved chunks generated from the
  // currently displayed Markdown variant — not every chunks file the document
  // has on disk.  Filter by md_source against the source we already derived
  // for useChunks above.
  const chunksForCurrentMd = useMemo(() => {
    if (currentMdSource === null) return availableChunks
    // Legacy chunk files that don't encode an md_source are kept visible so
    // pre-existing data isn't hidden after the upgrade.
    return availableChunks.filter(c => c.md_source == null || c.md_source === currentMdSource)
  }, [availableChunks, currentMdSource])

  // ── Panel-header version pickers ─────────────────────────────────────────
  // Rendered inside the .panel-label row alongside the PDF/Markdown/Chunks
  // tabs.  Lifted out of MarkdownViewer/ChunkViewer so they always have room
  // to coexist with the action toolbars and so their styling can match the
  // tabs (one shared toolbar feel).
  const renderMarkdownVersionPicker = (): React.ReactNode => {
    if (!availableMarkdowns || availableMarkdowns.length === 0) return null
    return (
      <select
        className="panel-version-select"
        value={selectedMarkdown ?? ''}
        onChange={e => {
          const v = availableMarkdowns.find(x => x.filename === e.target.value)
          if (!v) return
          // Identifier is the converter name when known, otherwise the raw
          // filename — matches the dual-lookup the backend supports.
          const id = v.source === 'converted' && v.converter ? v.converter : v.filename
          handleSelectMarkdownVersion(id)
        }}
        title="Switch between Markdown versions for this document"
      >
        {availableMarkdowns.map(v => {
          const baseLabel = v.source === 'converted' && v.converter
            ? v.converter
            : `${v.filename} · uploaded`
          // Append a ⚠ inline when this specific variant carries VLM
          // failure placeholders — surfaced from the backend so the marker
          // is consistent with the sidebar warning and the in-viewer
          // banner.  ``<option>`` text only, no styling: <select> elements
          // can't render rich content cross-browser.
          const label = v.has_failures ? `${baseLabel} ⚠` : baseLabel
          return <option key={v.filename} value={v.filename}>{label}</option>
        })}
      </select>
    )
  }

  /**
   * Compact zoom widget rendered inside the panel header.  Same scale
   * state for both PDF and MD viewers, just bound to a different setter
   * per panel.
   */
  const renderZoomControl = (
    scale: number,
    setScale: (s: number) => void,
  ): React.ReactNode => (
    <div className="panel-zoom" title="Zoom">
      <button onClick={() => setScale(Math.max(0.5, scale - 0.1))} disabled={scale <= 0.5}>−</button>
      <span>{(scale * 100).toFixed(0)}%</span>
      <button onClick={() => setScale(Math.min(3, scale + 0.1))} disabled={scale >= 3}>+</button>
    </div>
  )

  /**
   * View-options popover trigger for the Markdown viewer.  Currently only
   * exposes the padding slider; PDF doesn't need it so the trigger is
   * MD-only.
   */
  const renderMdOptionsTrigger = (panel: 'left' | 'right'): React.ReactNode => (
    <button
      className="panel-options-btn"
      onClick={() => setOptionsOpenIn(prev => (prev === panel ? null : panel))}
      title="View options"
    >⚙</button>
  )

  const renderMdOptionsPopover = (panel: 'left' | 'right'): React.ReactNode => (
    optionsOpenIn === panel
      ? (
          <div className="panel-options-popover" onClick={e => e.stopPropagation()}>
            <label>
              <span>Padding</span>
              <span>{mdPadding}px</span>
            </label>
            <input
              type="range"
              min={0}
              max={100}
              value={mdPadding}
              onChange={e => setMdPadding(+e.target.value)}
            />
          </div>
        )
      : null
  )

  const renderChunksVersionPicker = (): React.ReactNode => {
    if (!chunksForCurrentMd || chunksForCurrentMd.length === 0) return null
    return (
      <select
        className="panel-version-select"
        value={selectedChunks ?? ''}
        onChange={e => { if (e.target.value) handleLoadSavedChunks(e.target.value) }}
        title="Load a previously saved chunk version"
      >
        {!selectedChunks && <option value="" disabled hidden>— select a saved version —</option>}
        {chunksForCurrentMd.map(v => {
          const params: string[] = []
          if (v.chunk_size != null) params.push(`size ${v.chunk_size}`)
          if (v.chunk_overlap != null) params.push(`overlap ${v.chunk_overlap}`)
          const suffix = params.length > 0 ? ` (${params.join(', ')})` : ''
          return (
            <option key={v.filename} value={v.filename}>
              {`${v.library}/${v.algorithm}${suffix}`}
            </option>
          )
        })}
      </select>
    )
  }


  // ── Markdown panel helper — avoids duplicating MarkdownViewer props ──────
  const renderMarkdownPanel = (): React.ReactNode => {
    if (!documentData?.has_markdown) {
      return (
        <div className="md-not-found">
          <span className="static-icon">📄</span>
          <h2>Markdown not found</h2>
          <p>This document hasn't been converted yet.</p>
          <button onClick={handleConvert} disabled={converting}>
            ✨ Convert with {converterLabel}
          </button>
        </div>
      )
    }
    return (
      <MarkdownViewer
        content={documentData.md_content}
        scale={mdScale}
        padding={mdPadding}
        scrollSyncEnabled={scrollSync}
        onSaveMarkdown={saveMarkdown}
        onDeleteMarkdown={handleDeleteMarkdown}
        onConvert={handleConvert}
        converterLabel={converterLabel}
        activeConverter={settings.converter}
        converting={converting}
        savingMd={savingMd}
        sectionEnrichment={settings.sectionEnrichment}
        onEnrichSuccess={toastCallbacks.onSuccess}
        onEnrichError={toastCallbacks.onError}
        mdFilename={documentData.md_filename}
      />
    )
  }

  return (
    <div className="app">
      {toast && (
        <Toast
          key={toast.id}
          message={toast.message}
          type={toast.type}
          duration={toast.type === 'error' ? TOAST_DURATION_ERROR_MS : TOAST_DURATION_SUCCESS_MS}
          onClose={handleToastClose}
        />
      )}

      {/* ── Operation progress modals ── */}
      <ProgressModal
        isOpen={converting}
        title="PDF → Markdown"
        detail={conversionDetail}
        current={conversionProgress?.current ?? 0}
        total={conversionProgress?.total ?? 0}
        onInterrupt={cancelConversion}
        errorMessage={conversionErrorMessage ?? undefined}
      />
      <ProgressModal
        isOpen={chunking}
        title="Chunking Document"
        detail={selectedDoc ?? ''}
        current={0}
        total={0}
        onInterrupt={cancelChunking}
      />
      {bulkOp && (
        <ProgressModal
          isOpen
          title={bulkOp.title}
          detail={bulkOp.detail}
          current={bulkOp.current}
          total={bulkOp.total}
          onInterrupt={interruptBulk}
          errorMessage={bulkConnectionLost
            ? 'Connection lost — the operation may have been interrupted. You can safely start a new conversion.'
            : undefined}
        />
      )}

      {/* ── Unsaved chunks confirmation ── */}
      <ConfirmDialog
        isOpen={!!pendingChunkAction}
        onDismiss={handleCancelDirty}
        actions={[
          { label: 'Cancel', onClick: handleCancelDirty },
          { label: 'Discard', onClick: handleDiscardDirty },
          {
            label: savingChunks ? 'Saving…' : 'Save',
            onClick: handleSaveDirty,
            variant: 'primary',
            disabled: savingChunks,
          },
        ]}
      >
        <p>You have unsaved chunk changes. Save them before continuing?</p>
      </ConfirmDialog>

      {/* ── Confirm switch while processing ── */}
      <ConfirmDialog
        isOpen={!!pendingDoc}
        onDismiss={cancelSwitch}
        actions={[
          { label: 'Stay', onClick: cancelSwitch },
          { label: 'Switch document', onClick: confirmSwitch, variant: 'danger' },
        ]}
      >
        <p>
          A {converting || convertingToPdf ? 'conversion' : 'chunking'} is in progress.
          Switching documents will cancel it. Continue?
        </p>
      </ConfirmDialog>

      <Sidebar
        documents={documents}
        selectedDoc={selectedDoc}
        onSelect={handleSelectDocument}
        onUpload={uploadFiles}
        uploading={uploading}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(v => !v)}
        onDelete={deleteDocuments}
        onBulkConvert={handleBulkConvert}
        onBulkChunk={handleBulkChunk}
        onBulkEnrich={handleBulkEnrich}
        onOpenSettings={() => setSettingsOpen(true)}
        docsWithMarkdown={docsWithMarkdown}
        docsWithFailures={docsWithFailures}
      />

      <div className="main-content">
        {loading && <div className="loading-overlay" />}

        {!loading && !selectedDoc && (
          <div className="placeholder">
            <span>📄</span>
            <p>Select a document to get started</p>
          </div>
        )}

        {!loading && selectedDoc && documentData && (
          <>
            {/* ErrorBoundary resets when the selected document changes so a bad
                document doesn't permanently break the viewer. */}
            <ErrorBoundary key={selectedDoc}>
              <div className="viewers">
                {/* ── Left panel ── */}
                <div ref={leftPanelRef} className="viewer-panel" style={{ width: `${splitPct}%`, maxWidth: 'calc(100% - 8px)' }}>
                  <div className="panel-label">
                    <button
                      className={`panel-view-tab${leftView === 'pdf' ? ' active' : ''}`}
                      onClick={() => handleSetLeftView('pdf')}
                    >PDF</button>
                    <button
                      className={`panel-view-tab${leftView === 'markdown' ? ' active' : ''}`}
                      onClick={() => handleSetLeftView('markdown')}
                    >Markdown</button>
                    <div className="panel-label-tools">
                      {leftView === 'markdown' && documentData.has_markdown && renderMarkdownVersionPicker()}
                      {leftView === 'pdf' && documentData.has_pdf && renderZoomControl(pdfScale, setPdfScale)}
                      {leftView === 'markdown' && documentData.has_markdown && renderZoomControl(mdScale, setMdScale)}
                      {leftView === 'markdown' && documentData.has_markdown && renderMdOptionsTrigger('left')}
                    </div>
                  </div>
                  {renderMdOptionsPopover('left')}
                  {leftView === 'pdf' ? (
                    documentData.has_pdf ? (
                      <PDFViewer
                        filename={selectedDoc}
                        scale={pdfScale}
                        scrollSyncEnabled={scrollSync}
                        onToggleScrollSync={handleToggleScrollSync}
                      />
                    ) : (
                      <div className="md-not-found">
                        {convertingToPdf ? (
                          <>
                            <span className="hourglass-icon">⏳</span>
                            <h2>Converting to PDF…</h2>
                            <p>Please wait while we generate your PDF.</p>
                            <div className="converting-bar" />
                            <button className="btn-cancel-op" onClick={cancelMdToPdfConversion}>✕ Cancel</button>
                          </>
                        ) : (
                          <>
                            <span className="static-icon">📝</span>
                            <h2>No PDF available</h2>
                            <p>This document only has a Markdown file.</p>
                            <button onClick={convertMdToPdf}>✨ Convert to PDF</button>
                          </>
                        )}
                      </div>
                    )
                  ) : (
                    renderMarkdownPanel()
                  )}
                </div>

                {/* ── Divider ── */}
                <div
                  className="viewer-divider"
                  onMouseDown={() => setIsDragging(true)}
                  title="Drag to resize"
                >
                  <div className="viewer-divider-grip">
                    <span /><span />
                    <span /><span />
                    <span /><span />
                  </div>
                  <div className="viewer-divider-presets">
                    <div className="viewer-divider-presets-inner">
                      <button onClick={e => { e.stopPropagation(); setSplitPct(0) }} title="0 / 100">0·100</button>
                      <button onClick={e => { e.stopPropagation(); setSplitPct(40) }} title="40 / 60">40·60</button>
                      <button onClick={e => { e.stopPropagation(); setSplitPct(50) }} title="50 / 50">50·50</button>
                      <button onClick={e => { e.stopPropagation(); setSplitPct(60) }} title="60 / 40">60·40</button>
                      <button onClick={e => { e.stopPropagation(); setSplitPct(100) }} title="100 / 0">100·0</button>
                    </div>
                  </div>
                </div>

                {/* ── Right panel ── */}
                <div ref={rightPanelRef} className="viewer-panel" style={{ flex: '1 1 0', minWidth: 0 }}>
                  <div className="panel-label">
                    <button
                      className={`panel-view-tab${rightView === 'markdown' ? ' active' : ''}`}
                      onClick={() => handleSetRightView('markdown')}
                    >Markdown</button>
                    {documentData.has_markdown && (
                      <button
                        className={`panel-view-tab${rightView === 'chunks' ? ' active' : ''}`}
                        onClick={() => handleSetRightView('chunks')}
                      >Chunks</button>
                    )}
                    <div className="panel-label-tools">
                      {rightView === 'markdown' && documentData.has_markdown && renderMarkdownVersionPicker()}
                      {rightView === 'chunks' && renderChunksVersionPicker()}
                      {rightView === 'markdown' && documentData.has_markdown && renderZoomControl(mdScale, setMdScale)}
                      {rightView === 'markdown' && documentData.has_markdown && renderMdOptionsTrigger('right')}
                    </div>
                  </div>
                  {renderMdOptionsPopover('right')}
                  {rightView === 'markdown' ? (
                    renderMarkdownPanel()
                  ) : (
                    <ChunkViewer
                      chunks={chunks}
                      content={documentData.md_content}
                      chunksReady={!!chunks}
                      chunking={chunking}
                      savingChunks={savingChunks}
                      chunkEnrichment={settings.chunkEnrichment}
                      mdFilename={documentData.md_filename}
                      onEnrichChunk={enrichChunk}
                      onChunkEdit={editChunk}
                      onDeleteChunk={deleteChunk}
                      onDeleteChunks={deleteChunks}
                      onMergeChunks={mergeChunks}
                      onSaveChunks={handleSaveChunks}
                      onRechunk={handleRechunk}
                      chunkerLabel={settings.chunkerType}
                      scrollSyncEnabled={scrollSync}
                      onEnrichSuccess={toastCallbacks.onSuccess}
                      onEnrichError={toastCallbacks.onError}
                    />
                  )}
                </div>
              </div>
            </ErrorBoundary>
          </>
        )}

        <SettingsModal
          isOpen={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          onSave={handleApplySettings}
          current={settings}
        />
      </div>
    </div>
  )
}
