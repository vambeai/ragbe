import { useState, useRef } from 'react'
import { createPortal } from 'react-dom'
import type { BulkProgressFn, BulkResultFn } from '../../hooks/useDocument'
import logoSrc from '../../assets/logo.png'
import './Sidebar.css'

const WARNING_TOOLTIP_TEXT =
  'Partial VLM conversion — some pages failed or the run was interrupted. Click Convert again to retry.'

interface Props {
  documents: string[]
  selectedDoc: string | null
  onSelect: (doc: string) => void
  onUpload: (files: File[]) => void
  uploading: boolean
  collapsed: boolean
  onToggleCollapse: () => void
  onDelete: (filenames: string[]) => Promise<void>
  onBulkConvert?: (filenames: string[], onProgress: BulkProgressFn, onResult: BulkResultFn) => Promise<void>
  onBulkChunk?: (filenames: string[], onProgress: BulkProgressFn, onResult: BulkResultFn) => Promise<void>
  /**
   * Optional bulk markdown enrichment.  Available when the sidebar is
   * given a handler; the corresponding button is hidden otherwise.
   * Selected files without any markdown variant are skipped at runtime
   * by the handler (mirrors the Chunk button's behaviour).
   */
  onBulkEnrich?: (filenames: string[], onProgress: BulkProgressFn, onResult: BulkResultFn) => Promise<void>
  onOpenSettings?: () => void
  docsWithMarkdown?: Set<string>
  /**
   * PDFs whose last VLM conversion ended with at least one failed page.
   * Rendered as a warning icon next to the document name so partial
   * conversions are visible without opening the file.
   */
  docsWithFailures?: Set<string>
}

interface BulkOpState {
  type: 'convert' | 'chunk' | 'enrich'
  current: number
  total: number
  currentFile: string
  results: Map<string, boolean>
}

export default function Sidebar({
  documents, selectedDoc, onSelect, onUpload, uploading,
  collapsed, onToggleCollapse, onDelete,
  onBulkConvert, onBulkChunk, onBulkEnrich,
  onOpenSettings, docsWithMarkdown, docsWithFailures,
}: Props) {
  const [search, setSearch] = useState('')
  const [selectMode, setSelectMode] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [deleting, setDeleting] = useState(false)
  const [bulkOp, setBulkOp] = useState<BulkOpState | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [warningTip, setWarningTip] = useState<{ x: number; y: number } | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dragDepthRef = useRef(0)

  const showWarningTip = (e: React.MouseEvent<HTMLSpanElement> | React.FocusEvent<HTMLSpanElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    setWarningTip({ x: r.right, y: r.top })
  }
  const hideWarningTip = () => setWarningTip(null)

  const filtered = documents.filter(d => d.toLowerCase().includes(search.toLowerCase()))

  const handleFiles = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? [])
    if (files.length > 0) onUpload(files)
    e.target.value = ''
  }

  const acceptFile = (f: File) => /\.(pdf|md)$/i.test(f.name)

  const handleDragEnter = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    e.stopPropagation()
    if (uploading) return
    dragDepthRef.current += 1
    setDragOver(true)
  }

  const handleDragOver = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    e.stopPropagation()
    if (uploading) {
      e.dataTransfer.dropEffect = 'none'
    } else {
      e.dataTransfer.dropEffect = 'copy'
    }
  }

  const handleDragLeave = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    e.stopPropagation()
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)
    if (dragDepthRef.current === 0) setDragOver(false)
  }

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    e.stopPropagation()
    dragDepthRef.current = 0
    setDragOver(false)
    if (uploading) return
    const files = Array.from(e.dataTransfer.files ?? []).filter(acceptFile)
    if (files.length > 0) onUpload(files)
  }

  const toggleSelectMode = () => {
    setSelectMode(v => !v)
    setSelected(new Set())
  }

  const toggleDoc = (doc: string) => {
    setSelected(prev => {
      const next = new Set(prev)
      next.has(doc) ? next.delete(doc) : next.add(doc)
      return next
    })
  }

  const handleDeleteSelected = async () => {
    if (selected.size === 0) return
    setDeleting(true)
    try {
      await onDelete(Array.from(selected))
      setSelected(new Set())
      setSelectMode(false)
    } finally {
      setDeleting(false)
    }
  }

  const handleDeleteSingle = async (e: React.MouseEvent, doc: string) => {
    e.stopPropagation()
    setDeleting(true)
    try {
      await onDelete([doc])
    } finally {
      setDeleting(false)
    }
  }

  const runBulkOp = async (type: BulkOpState['type'], handler: typeof onBulkConvert) => {
    if (!handler || selected.size === 0) return
    // Chunking and enrichment both require markdown — skip files
    // without any variant on disk.  Conversion takes the raw PDF
    // list as-is.
    const filenames = (type === 'chunk' || type === 'enrich')
      ? Array.from(selected).filter(f => docsWithMarkdown?.has(f))
      : Array.from(selected)
    if (filenames.length === 0) return
    const results = new Map<string, boolean>()

    setBulkOp({ type, current: 0, total: filenames.length, currentFile: '', results })

    const onProgress: BulkProgressFn = (current, total, filename) =>
      setBulkOp(prev => prev ? { ...prev, current, total, currentFile: filename } : prev)

    const onResult: BulkResultFn = (filename, success) => {
      results.set(filename, success)
      setBulkOp(prev => prev ? { ...prev, results } : prev)
    }

    try {
      await handler(filenames, onProgress, onResult)
    } finally {
      setBulkOp(null)
      setSelected(new Set())
      setSelectMode(false)
    }
  }

  const isBusy = deleting || bulkOp !== null

  return (
    <div className={`sidebar ${collapsed ? 'collapsed' : ''}`}>

      {/* ── Fixed top ── */}
      <div className="sidebar-fixed">

        <div className="sidebar-brand">
          <img src={logoSrc} alt="logo" className="sidebar-logo" />
          {!collapsed && <span className="sidebar-app-name">Chunky</span>}
        </div>

        {collapsed ? (
          <div className="sidebar-collapsed-toggle">
            <button className="menu-toggle" onClick={onToggleCollapse} title="Expand sidebar">☰</button>
          </div>
        ) : (
          <>
            <div className="sidebar-divider" />

            <div className="sidebar-section-row">
              <button className="menu-toggle" onClick={onToggleCollapse} title="Collapse sidebar">☰</button>
              <span className="sidebar-section-label">Documents</span>
            </div>

            <div className="sidebar-upload">
              <input
                ref={fileInputRef}
                type="file"
                id="file-upload"
                accept=".pdf,.md"
                multiple
                onChange={handleFiles}
                style={{ display: 'none' }}
              />
              <label
                htmlFor="file-upload"
                className={`upload-btn${dragOver ? ' drag-over' : ''}`}
                onDragEnter={handleDragEnter}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                {uploading
                  ? <><span>⏳</span> Uploading…</>
                  : dragOver
                    ? <><span>📥</span> Drop to upload</>
                    : <><span>📤</span> Upload PDF / MD</>}
              </label>
            </div>

            <div className="sidebar-search">
              <input
                type="text"
                placeholder="Search documents…"
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="search-input"
              />
            </div>

            <div className="doc-count-row">
              <span className="doc-count">
                {filtered.length} / {documents.length}{' '}
                {documents.length === 1 ? 'document' : 'documents'}
              </span>
              {documents.length > 0 && (
                <button
                  className={`select-toggle-btn ${selectMode ? 'active' : ''}`}
                  onClick={toggleSelectMode}
                  title={selectMode ? 'Cancel selection' : 'Select documents'}
                  disabled={isBusy}
                >
                  {selectMode ? '✕' : 'Select'}
                </button>
              )}
            </div>

            {selectMode && (
              <>
                <div className="bulk-actions">
                  <button
                    className="bulk-btn select-all"
                    onClick={() => setSelected(new Set(filtered))}
                    disabled={selected.size === filtered.length || isBusy}
                  >
                    Select all
                  </button>
                  <button
                    className="bulk-btn delete-selected"
                    onClick={handleDeleteSelected}
                    disabled={selected.size === 0 || isBusy}
                  >
                    {deleting
                      ? '⏳ Deleting…'
                      : `🗑 Delete${selected.size > 0 ? ` (${selected.size})` : ''}`}
                  </button>
                </div>

                {(onBulkConvert || onBulkChunk || onBulkEnrich) && (
                  <div className="bulk-actions bulk-actions-secondary">
                    {onBulkConvert && (
                      <button
                        className="bulk-btn convert-selected"
                        onClick={() => runBulkOp('convert', onBulkConvert)}
                        disabled={selected.size === 0 || isBusy}
                      >
                        {bulkOp?.type === 'convert'
                          ? `⏳ ${bulkOp.current}/${bulkOp.total}`
                          : `✨ Convert${selected.size > 0 ? ` (${selected.size})` : ''}`}
                      </button>
                    )}
                    {onBulkChunk && (() => {
                      const chunkable = Array.from(selected).filter(f => docsWithMarkdown?.has(f)).length
                      return (
                        <button
                          className="bulk-btn chunk-selected"
                          onClick={() => runBulkOp('chunk', onBulkChunk)}
                          disabled={chunkable === 0 || isBusy}
                          title={selected.size > chunkable ? `${selected.size - chunkable} selected file(s) have no markdown and will be skipped` : undefined}
                        >
                          {bulkOp?.type === 'chunk'
                            ? `⏳ ${bulkOp.current}/${bulkOp.total}`
                            : `⛓ Chunk${chunkable > 0 ? ` (${chunkable})` : ''}`}
                        </button>
                      )
                    })()}
                    {onBulkEnrich && (() => {
                      const enrichable = Array.from(selected).filter(f => docsWithMarkdown?.has(f)).length
                      return (
                        <button
                          className="bulk-btn enrich-selected"
                          onClick={() => runBulkOp('enrich', onBulkEnrich)}
                          disabled={enrichable === 0 || isBusy}
                          title={selected.size > enrichable ? `${selected.size - enrichable} selected file(s) have no markdown and will be skipped` : 'Bulk enrichment skips the summary review modal and uses any cached summary silently.'}
                        >
                          {bulkOp?.type === 'enrich'
                            ? `⏳ ${bulkOp.current}/${bulkOp.total}`
                            : `✨ Enrich${enrichable > 0 ? ` (${enrichable})` : ''}`}
                        </button>
                      )
                    })()}
                  </div>
                )}

                {bulkOp && (
                  <div className="bulk-progress">
                    <span className="bulk-progress-label">
                      {bulkOp.type === 'convert' ? 'Converting' :
                        bulkOp.type === 'chunk' ? 'Chunking' : 'Enriching'}{' '}
                      {bulkOp.current}/{bulkOp.total}
                    </span>
                    {bulkOp.currentFile && (
                      <span className="bulk-progress-file" title={bulkOp.currentFile}>
                        {bulkOp.currentFile}
                      </span>
                    )}
                    <div className="bulk-progress-results">
                      {Array.from(bulkOp.results.entries()).map(([file, ok]) => (
                        <span
                          key={file}
                          className={`bulk-result-dot ${ok ? 'ok' : 'fail'}`}
                          title={`${file}: ${ok ? 'success' : 'failed'}`}
                        />
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>

      {/* ── Scrollable list ── */}
      {!collapsed && (
        <div className="sidebar-list-scroll">
          <ul className="doc-list">
            {filtered.length === 0
              ? <li className="no-docs">No results</li>
              : filtered.map(doc => {
                  const isSelected = selected.has(doc)
                  const isActive = selectedDoc === doc && !selectMode
                  return (
                    <li
                      key={doc}
                      className={[
                        isActive ? 'active' : '',
                        selectMode && isSelected ? 'selected' : '',
                      ].filter(Boolean).join(' ')}
                      onClick={() => selectMode ? toggleDoc(doc) : onSelect(doc)}
                      title={doc}
                    >
                      {selectMode && (
                        <span className={`doc-checkbox ${isSelected ? 'checked' : ''}`} />
                      )}
                      <span className="doc-icon">📄</span>
                      <span className="doc-name">{doc}</span>
                      {docsWithFailures?.has(doc) && (
                        <span
                          className="doc-warning-badge"
                          aria-label={WARNING_TOOLTIP_TEXT}
                          tabIndex={0}
                          onMouseEnter={showWarningTip}
                          onMouseLeave={hideWarningTip}
                          onFocus={showWarningTip}
                          onBlur={hideWarningTip}
                        >
                          {/* Inline SVG warning triangle.  Renders identically
                              across platforms (no Unicode/emoji font roulette),
                              inherits the colour via ``currentColor``, and
                              matches the stroke style used elsewhere in the
                              app (see the password-toggle icons). */}
                          <svg
                            width="14"
                            height="14"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            aria-hidden="true"
                          >
                            <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
                            <line x1="12" y1="9" x2="12" y2="13" />
                            <line x1="12" y1="17" x2="12.01" y2="17" />
                          </svg>
                        </span>
                      )}
                      {docsWithMarkdown?.has(doc) && (
                        <span className="doc-md-badge" title="Markdown available">MD</span>
                      )}
                      {!selectMode && (
                        <button
                          className="doc-delete-btn"
                          onClick={e => handleDeleteSingle(e, doc)}
                          title={`Delete ${doc}`}
                          disabled={deleting}
                        >
                          🗑
                        </button>
                      )}
                    </li>
                  )
                })
            }
          </ul>
        </div>
      )}

      {/* ── Settings button at bottom ── */}
      {onOpenSettings && (
        <div className={`sidebar-settings-footer ${collapsed ? 'collapsed' : ''}`}>
          <button
            className="sidebar-settings-btn"
            onClick={onOpenSettings}
            title="Settings"
          >
            <span className="sidebar-settings-icon">⚙️</span>
            {!collapsed && <span className="sidebar-settings-label">Settings</span>}
          </button>
        </div>
      )}

      {warningTip && createPortal(
        <div
          className="doc-warning-tooltip"
          role="tooltip"
          style={{ left: warningTip.x + 8, top: warningTip.y }}
        >
          {WARNING_TOOLTIP_TEXT}
        </div>,
        document.body,
      )}
    </div>
  )
}
