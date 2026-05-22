import { useRef, useEffect, useState, useMemo, memo, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { EnrichmentSettings } from '../../types'
import { useScrollSync } from '../../hooks/useScrollSync'
import { VIEWER_PAGE_SYNC } from '../../utils/viewerEvents'
import { LAZY_OBSERVER_MARGIN, LAZY_SECTION_TARGET_CHARS } from '../../config'
import ProgressModal from '../modals/ProgressModal'
import EnrichDiffModal from '../modals/EnrichDiffModal'
import SummaryReviewModal from '../modals/SummaryReviewModal'
import { apiEnrichMarkdownPipeline, type PipelineProgress, type PipelineResult } from '../../services/apiService'
import { missingEnrichmentModelError } from '../../utils/chunkUtils'
import './MarkdownViewer.css'

interface Props {
  content: string
  /** Display scale (managed and rendered by the parent — see App.tsx). */
  scale?: number
  /** Padding around the rendered Markdown (managed by the parent). */
  padding?: number
  scrollSyncEnabled?: boolean
  onSaveMarkdown: (content: string) => Promise<void>
  onDeleteMarkdown: () => void
  /** Triggers PDF→Markdown conversion with the converter currently selected
   *  in Settings.  The backend short-circuits when the resulting file
   *  already exists, so re-clicking with the same converter is harmless. */
  onConvert: () => void
  /** User-facing label for the active converter (shown on the Convert
   *  button so the user knows which method will run). */
  converterLabel: string
  /** Canonical converter id ("vlm", "pymupdf", …) — used to gate
   *  VLM-specific affordances like the "Retry failed pages" action. */
  activeConverter: string
  converting: boolean
  savingMd: boolean
  sectionEnrichment?: EnrichmentSettings
  onEnrichSuccess?: (msg: string) => void
  onEnrichError?: (msg: string) => void
  /** Filename of the markdown variant currently displayed.  Required by
   *  the pipeline enrichment endpoint — the backend loads the source
   *  content from disk so the checkpoint key stays stable across
   *  requests. */
  mdFilename?: string
}

// ── Page-marker helpers ─────────────────────────────────────────────────────

// Split content at page-marker comments into per-page sections.
// This avoids false positives from `---` horizontal rules inside page content.
function splitAtPageMarkers(md: string): Array<{ page: number; content: string }> | null {
  const parts = md.split(/<!--\s*page-marker:(\d+)\s*-->/)
  // parts = [before_first_marker, '1', content1, '2', content2, ...]
  if (parts.length < 3) return null
  const sections: Array<{ page: number; content: string }> = []
  for (let i = 1; i < parts.length; i += 2) {
    const page = parseInt(parts[i], 10)
    // Strip the trailing `\n\n---\n\n` page-separator (added by the backend between pages)
    const content = (parts[i + 1] ?? '').replace(/\n\n---\n\n$/, '').trim()
    sections.push({ page, content })
  }
  return sections.length > 0 ? sections : null
}

// ── Lazy section splitting ──────────────────────────────────────────────────

// Split markdown at blank-line boundaries into groups of ~targetSize chars.
// Splitting at \n\n guarantees we never cut inside a block element (table row,
// code fence, list item) so each section is always valid standalone markdown.
function splitIntoLazySections(md: string, targetSize = LAZY_SECTION_TARGET_CHARS): string[] {
  const blocks = md.split(/\n\n+/)
  const sections: string[] = []
  let current = ''
  for (const block of blocks) {
    if (current.length > 0 && current.length + block.length + 2 > targetSize) {
      sections.push(current)
      current = block
    } else {
      current = current.length > 0 ? `${current}\n\n${block}` : block
    }
  }
  if (current.length > 0) sections.push(current)
  return sections.length > 0 ? sections : [md]
}

// Renders a height-matched placeholder until the section scrolls within 300 px
// of the viewport, then replaces it with the parsed ReactMarkdown output.
// React.memo prevents re-parsing when the parent re-renders with the same content.
const LazySection = memo(function LazySection({
  content: sectionContent,
  estimatedHeight,
}: {
  content: string
  estimatedHeight: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) { setVisible(true); observer.disconnect() }
      },
      { rootMargin: LAZY_OBSERVER_MARGIN },
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  if (visible) return <ReactMarkdown remarkPlugins={[remarkGfm]}>{sectionContent}</ReactMarkdown>
  return <div ref={ref} style={{ minHeight: estimatedHeight }} />
})

// ── Component ──────────────────────────────────────────────────────────────

function MarkdownViewer({
  content, scale = 1.0, padding = 20,
  scrollSyncEnabled = true,
  onSaveMarkdown, onDeleteMarkdown,
  onConvert, converterLabel, activeConverter, converting,
  savingMd,
  sectionEnrichment,
  onEnrichSuccess, onEnrichError,
  mdFilename,
}: Props) {
  const [editMode, setEditMode] = useState(false)
  const [editContent, setEditContent] = useState(content)
  const [enrichError, setEnrichError] = useState<string | null>(null)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)

  // ── Pipeline enrichment state ────────────────────────────────────────────
  // Streams `cleanup_done` → `split_done` → `piece_done` × N → `done`.
  // The diff modal appears after `done`; the user accepts (writes to disk)
  // or rejects (discards the enriched output, keeps original).
  const [pipelineProgress, setPipelineProgress] = useState<PipelineProgress | null>(null)
  const [pipelineResult, setPipelineResult] = useState<PipelineResult | null>(null)
  const [pipelineSaving, setPipelineSaving] = useState(false)
  const pipelineAbortRef = useRef<AbortController | null>(null)
  useEffect(() => () => { pipelineAbortRef.current?.abort() }, [])

  // Enrich is a two-step flow: clicking the button (usually) opens the
  // SummaryReviewModal where the user inspects / edits / regenerates the
  // per-document summary.  Only after they click "Confirm & enrich" does
  // the actual pipeline kick off.  The modal handles its own PUT to
  // persist edits before this runs, so by the time we get here the
  // summary on disk is the one the user approved.
  //
  // The Settings → "Skip document summary" checkbox short-circuits this
  // flow: when on, clicking Enrich runs the pipeline directly with the
  // summary disabled (use_summary=false on the request).  No modal.
  const [summaryModalOpen, setSummaryModalOpen] = useState(false)

  const handleOpenEnrich = useCallback(() => {
    if (!sectionEnrichment?.model) {
      setEnrichError(missingEnrichmentModelError('Section Enrichment'))
      return
    }
    if (!mdFilename) {
      setEnrichError('No Markdown file selected.')
      return
    }
    setEnrichError(null)
    if (sectionEnrichment.skip_summary) {
      // Setting overrides the per-run modal entirely.
      startPipelineAfterReview(false)
      return
    }
    setSummaryModalOpen(true)
  }, [sectionEnrichment, mdFilename])

  const startPipelineAfterReview = useCallback(async (useSummary: boolean) => {
    if (!mdFilename || !sectionEnrichment) return
    pipelineAbortRef.current?.abort()
    const ctrl = new AbortController()
    pipelineAbortRef.current = ctrl

    setPipelineProgress({
      totalPieces: 0,
      completedPieces: 0,
      inFlight: 0,
      cachedPieces: 0,
      failedPieces: [],
    })

    const enrichSettings: EnrichmentSettings = sectionEnrichment
    try {
      const useCheckpoint = enrichSettings.use_checkpoint ?? true
      const result = await apiEnrichMarkdownPipeline(
        mdFilename,
        enrichSettings,
        useCheckpoint,
        useSummary,
        progress => setPipelineProgress(progress),
        ctrl.signal,
        () => { /* connection-lost: parseSse aborts the loop; we surface below */ },
      )
      // Always show the diff modal — even when the document is byte-identical
      // (no changes).  That way the user gets explicit confirmation that the
      // pipeline ran and found nothing to fix, instead of a silent no-op.
      setPipelineResult(result)
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        // user-initiated cancel — silent
      } else {
        onEnrichError?.(err instanceof Error ? err.message : 'Pipeline enrichment failed')
      }
    } finally {
      if (pipelineAbortRef.current === ctrl) {
        pipelineAbortRef.current = null
        setPipelineProgress(null)
      }
    }
  }, [sectionEnrichment, mdFilename, onEnrichError])

  const handleConfirmSummary = useCallback((useSummary: boolean) => {
    // Modal already PUT any edits before invoking this callback, so we
    // just need to fire the enrichment pipeline.  ``useSummary`` echoes
    // whichever button the user clicked: ``true`` for Confirm & enrich
    // (the pipeline loads the persisted summary), ``false`` for Skip
    // in the empty state (the pipeline runs without any summary).
    startPipelineAfterReview(useSummary)
  }, [startPipelineAfterReview])

  const handleInterruptPipeline = useCallback(() => {
    pipelineAbortRef.current?.abort()
    setPipelineProgress(null)
  }, [])

  // The modal owns the (possibly user-edited) buffer and hands it to us
  // when the user clicks Accept.  Saving uses that buffer, not the raw
  // pipeline output, so manual edits in the diff modal are persisted.
  const handleAcceptPipeline = useCallback(async (content: string) => {
    setPipelineSaving(true)
    try {
      await onSaveMarkdown(content)
      onEnrichSuccess?.('Enrichment applied ✓')
      setPipelineResult(null)
    } catch {
      onEnrichError?.('Failed to save enriched markdown')
    } finally {
      setPipelineSaving(false)
    }
  }, [onSaveMarkdown, onEnrichSuccess, onEnrichError])

  const handleRejectPipeline = useCallback(() => {
    setPipelineResult(null)
  }, [])

  // The progress modal's "current/total" needs a sensible value before the
  // splitter finishes: while we only know we're in the cleanup stage, show
  // an indeterminate bar (total=0).  After split_done we have the real total.
  // The pipeline no longer builds the summary inline — that runs from the
  // SummaryReviewModal before this even starts — so progress here is just
  // cleanup → per-piece corrections.
  const pipelineCurrent = pipelineProgress?.completedPieces ?? 0
  const pipelineTotal = pipelineProgress?.totalPieces ?? 0
  // Detail text reports the number of pieces *finished*, not pieces started.
  // The backend emits a ``piece_start`` event the moment each task spins up,
  // and all N tasks are created at once before the first one acquires the
  // concurrency semaphore — so a naive "started so far" counter races to N
  // immediately and freezes there.  Using ``completedPieces`` instead means
  // the number rises in lockstep with the progress bar.
  const pipelineDetail = pipelineProgress
    ? pipelineTotal === 0
      ? 'Cleaning markdown…'
      : `${pipelineCurrent} of ${pipelineTotal} piece${pipelineTotal === 1 ? '' : 's'} corrected` +
        (pipelineProgress.cachedPieces > 0 ? ` · ${pipelineProgress.cachedPieces} from cache` : '') +
        (pipelineProgress.failedPieces.length > 0 ? ` · ${pipelineProgress.failedPieces.length} failed` : '')
    : ''

  const containerRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const restoreRafRef = useRef<number>()
  const savedScrollRatioRef = useRef<number>(0)

  // Pre-compute lazy sections once per content change.
  // For VLM output (page markers present) each page is further split into
  // ~2 500-char buckets. For plain output the whole string is bucketed directly.
  // Either way, only sections near the viewport are ever parsed by ReactMarkdown.
  const lazyContent = useMemo(() => {
    const paged = splitAtPageMarkers(content)
    if (paged) {
      return {
        type: 'paged' as const,
        pages: paged.map(ps => ({
          page: ps.page,
          sections: splitIntoLazySections(ps.content),
        })),
      }
    }
    return { type: 'single' as const, sections: splitIntoLazySections(content) }
  }, [content])

  const hasPageSync = lazyContent.type === 'paged'

  // ── Failed-page detection ────────────────────────────────────────────────
  // The VLM converter emits ``<!-- page N failed: <error> -->`` placeholders
  // for pages that could not be transcribed.  Surface these to the user
  // with a banner + a retry button (which re-runs the conversion; the
  // backend's checkpoint logic skips successful pages automatically).
  const failedPages = useMemo<number[]>(() => {
    const re = /<!--\s*page\s+(\d+)\s+failed:/g
    const out: number[] = []
    let m: RegExpExecArray | null
    while ((m = re.exec(content)) !== null) {
      const n = parseInt(m[1], 10)
      if (!Number.isNaN(n)) out.push(n)
    }
    return Array.from(new Set(out)).sort((a, b) => a - b)
  }, [content])

  const formatFailedPages = (pages: number[]): string => {
    if (pages.length <= 8) return pages.join(', ')
    return `${pages.slice(0, 6).join(', ')}, … (+${pages.length - 6} more)`
  }

  // Listen for page-sync events from the PDF viewer.
  useEffect(() => {
    if (!hasPageSync) return
    const scrollToPage = (pageNum: number) => {
      if (!containerRef.current) return
      const anchor = document.getElementById(`md-page-anchor-${pageNum}`)
      if (!anchor) {
        if (pageNum === 1) containerRef.current.scrollTo({ top: 0, behavior: 'smooth' })
        return
      }
      const containerRect = containerRef.current.getBoundingClientRect()
      const anchorRect = anchor.getBoundingClientRect()
      const scrollTop = containerRef.current.scrollTop + (anchorRect.top - containerRect.top) - 8
      containerRef.current.scrollTo({ top: Math.max(0, scrollTop), behavior: 'smooth' })
    }
    const handler = (e: Event) => {
      const ev = e as CustomEvent
      if (ev.detail.source !== 'pdf') return
      scrollToPage(ev.detail.page as number)
    }
    window.addEventListener(VIEWER_PAGE_SYNC, handler)
    return () => window.removeEventListener(VIEWER_PAGE_SYNC, handler)
  }, [hasPageSync])

  useEffect(() => {
    setEditContent(content)
    setEditMode(false)
    setEnrichError(null)
  }, [content])

  // ── Scroll ratio save/restore ──────────────────────────────────────────────

  const restoreScrollRatio = (toEditMode: boolean) => {
    if (restoreRafRef.current !== undefined) cancelAnimationFrame(restoreRafRef.current)
    restoreRafRef.current = requestAnimationFrame(() => {
      restoreRafRef.current = undefined
      const el: HTMLElement | null = toEditMode
        ? (textareaRef.current ?? containerRef.current)
        : containerRef.current
      if (!el) return
      const max = el.scrollHeight - el.clientHeight
      if (max > 0) el.scrollTop = savedScrollRatioRef.current * max
    })
  }

  const handleEnterEdit = () => {
    const el = containerRef.current
    if (el) {
      const max = el.scrollHeight - el.clientHeight
      savedScrollRatioRef.current = max > 0 ? el.scrollTop / max : 0
    }
    setEditMode(true)
  }

  useEffect(() => {
    restoreScrollRatio(editMode)
    return () => { if (restoreRafRef.current !== undefined) cancelAnimationFrame(restoreRafRef.current) }
  }, [editMode])

  const handleSaveMd = async () => {
    const ta = textareaRef.current
    if (ta) {
      const max = ta.scrollHeight - ta.clientHeight
      savedScrollRatioRef.current = max > 0 ? ta.scrollTop / max : 0
    }
    await onSaveMarkdown(editContent)
    setEditMode(false)
  }

  const handleCancelEdit = () => {
    const ta = textareaRef.current
    if (ta) {
      const max = ta.scrollHeight - ta.clientHeight
      savedScrollRatioRef.current = max > 0 ? ta.scrollTop / max : 0
    }
    setEditContent(content)
    setEditMode(false)
    setEnrichError(null)
  }

  const handleConfirmDelete = () => {
    setShowDeleteConfirm(false)
    onDeleteMarkdown()
  }


  // ── Scroll sync ────────────────────────────────────────────────────────────

  const { handleScroll } = useScrollSync(
    scrollSyncEnabled ?? false,
    'markdown',
    'pdf',
    containerRef,
    (pct) => { savedScrollRatioRef.current = pct },
  )

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="md-viewer-wrapper">
      {/* Enrichment progress modal — driven by the pipeline endpoint
          (regex cleanup + structure-aware split + per-piece LLM with
          rolling context). */}
      <ProgressModal
        isOpen={pipelineProgress !== null}
        title="Enrichment"
        detail={pipelineDetail}
        current={pipelineCurrent}
        total={pipelineTotal}
        onInterrupt={handleInterruptPipeline}
      />

      {/* Diff preview — shown after the pipeline completes; the user
          must explicitly accept before the enriched content is written
          to disk. */}
      <EnrichDiffModal
        isOpen={pipelineResult !== null}
        original={editMode ? editContent : content}
        result={pipelineResult}
        onAccept={handleAcceptPipeline}
        onReject={handleRejectPipeline}
        onClose={handleRejectPipeline}
        saving={pipelineSaving}
      />

      {/* Summary Review modal — the first step of the Enrich flow.  Lets
          the user inspect, edit, or regenerate the per-document summary
          before any piece-correction LLM calls run.  Confirming starts
          the enrichment pipeline; cancelling closes without enriching. */}
      <SummaryReviewModal
        isOpen={summaryModalOpen}
        filename={mdFilename ?? null}
        settings={sectionEnrichment ?? {}}
        onClose={() => setSummaryModalOpen(false)}
        onConfirm={useSummary => {
          setSummaryModalOpen(false)
          handleConfirmSummary(useSummary)
        }}
      />

      {/* Delete confirmation dialog */}
      {showDeleteConfirm && (
        <div className="reconvert-confirm-overlay" onClick={() => setShowDeleteConfirm(false)}>
          <div className="reconvert-confirm" onClick={e => e.stopPropagation()}>
            <p>Delete the currently displayed Markdown variant? Other variants of this document stay intact.</p>
            <div className="reconvert-confirm-actions">
              <button className="btn-secondary" onClick={() => setShowDeleteConfirm(false)}>Cancel</button>
              <button className="btn-danger" onClick={handleConfirmDelete}>Delete</button>
            </div>
          </div>
        </div>
      )}

      {/* Primary controls bar — actions only.  Zoom + padding live in the
          panel header (see App.tsx) so this row never overflows. */}
      <div className="md-controls">
        <div className="md-controls-right">
          <button
            className="md-action-btn convert-now"
            onClick={onConvert}
            disabled={converting}
            title={`Convert this PDF with ${converterLabel}`}
          >
            <span>✨</span> Convert with {converterLabel}
          </button>
          <button
            className="md-action-btn delete-md"
            onClick={() => setShowDeleteConfirm(true)}
            title="Delete this Markdown variant"
          >
            <span>🗑</span> Delete
          </button>

          <div className="md-edit-actions">
            {!editMode ? (
              <>
                <button className="md-action-btn edit" onClick={handleEnterEdit}>
                  ✏️ Edit
                </button>
                <button
                  className="md-action-btn enrich"
                  onClick={handleOpenEnrich}
                  title="Review the document summary, then clean and LLM-correct the markdown piece-by-piece. Preview the diff before saving."
                  disabled={pipelineProgress !== null || pipelineResult !== null || summaryModalOpen}
                >
                  ✨ Enrich
                </button>
              </>
            ) : (
              <>
                <button
                  className="md-action-btn enrich"
                  onClick={handleOpenEnrich}
                  title="Review the document summary, then clean and LLM-correct the markdown piece-by-piece. Preview the diff before saving."
                  disabled={pipelineProgress !== null || pipelineResult !== null || summaryModalOpen}
                >
                  ✨ Enrich
                </button>
                <button className="md-action-btn save-md" onClick={handleSaveMd} disabled={savingMd}>
                  {savingMd ? '⏳ Saving…' : '💾 Save'}
                </button>
                <button className="md-action-btn cancel" onClick={handleCancelEdit}>✕ Cancel</button>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Enrich error banner */}
      {enrichError && (
        <div className="enrich-error-banner">
          ⚠️ {enrichError}
          <button className="enrich-error-close" onClick={() => setEnrichError(null)}>✕</button>
        </div>
      )}

      {/* VLM partial-conversion banner — surfaces failed pages and offers
          a one-click retry. Re-running the conversion is safe: the backend
          skips already-cached pages from the checkpoint and re-attempts
          only the failures. The retry button is gated on the active
          converter being VLM, otherwise clicking it would silently run a
          different converter (which couldn't reuse the VLM checkpoint).*/}
      {failedPages.length > 0 && (() => {
        const canRetry = activeConverter === 'vlm'
        return (
          <div className="failed-pages-banner">
            <span>
              ⚠️ {failedPages.length} page{failedPages.length !== 1 ? 's' : ''} failed transcription:
              {' '}<strong>{formatFailedPages(failedPages)}</strong>
            </span>
            <button
              className="failed-pages-retry"
              onClick={onConvert}
              disabled={converting || !canRetry}
              title={canRetry
                ? 'Re-run conversion. Successful pages are reused from the checkpoint; failed pages are retried.'
                : 'Switch the converter to VLM in Settings to retry failed pages.'}
            >
              ↻ Retry failed pages
            </button>
          </div>
        )
      })()}

      {/* Viewer / editor */}
      <div
        className="md-viewer"
        ref={containerRef}
        onScroll={handleScroll}
        style={{ fontSize: `${(11 * scale).toFixed(1)}pt` }}
      >
        {content ? (
          editMode ? (
            <textarea
              ref={textareaRef}
              className="md-raw-editor"
              value={editContent}
              onChange={e => setEditContent(e.target.value)}
              style={{ padding: `${padding}px` }}
              spellCheck={false}
            />
          ) : (
            <div className="markdown-content" style={{ padding: `${padding}px` }}>
              {lazyContent.type === 'paged' ? (
                lazyContent.pages.map(({ page, sections }) => (
                  <div key={page}>
                    {page === 1
                      ? <div id="md-page-anchor-1" style={{ height: 0 }} />
                      : (
                        <div className="md-page-break" id={`md-page-anchor-${page}`}>
                          <hr />
                          <button
                            className="md-page-label"
                            title={`Jump PDF to page ${page}`}
                            onClick={() => window.dispatchEvent(new CustomEvent(VIEWER_PAGE_SYNC, {
                              detail: { source: 'markdown', page },
                            }))}
                          >
                            Page {page}
                          </button>
                        </div>
                      )
                    }
                    {sections.map((section, i) => (
                      <LazySection
                        key={i}
                        content={section}
                        estimatedHeight={Math.max(80, section.length * 0.3)}
                      />
                    ))}
                  </div>
                ))
              ) : (
                lazyContent.sections.map((section, i) => (
                  <LazySection
                    key={i}
                    content={section}
                    estimatedHeight={Math.max(80, section.length * 0.3)}
                  />
                ))
              )}
            </div>
          )
        ) : (
          <div className="no-markdown"><p>No markdown content available</p></div>
        )}
      </div>
    </div>
  )
}

export default memo(MarkdownViewer)
