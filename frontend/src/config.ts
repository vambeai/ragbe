/**
 * Frontend-wide tunables.  Centralised here so values that affect UX, lazy
 * rendering, networking, or styling can be tweaked without grepping the
 * codebase.  Anything magic that appears in more than one component should
 * live in this file.
 */

// ── Networking ─────────────────────────────────────────────────────────────

/** Timeout for ad-hoc metadata fetches (e.g. /documents/metadata). */
export const METADATA_FETCH_TIMEOUT_MS = 5000

// ── Toast notifications ────────────────────────────────────────────────────

export const TOAST_DURATION_SUCCESS_MS = 4000
export const TOAST_DURATION_ERROR_MS = 10000

// ── Lazy rendering ─────────────────────────────────────────────────────────

/** rootMargin used by IntersectionObserver to pre-render content slightly
 *  outside the viewport so scrolling feels seamless. */
export const LAZY_OBSERVER_MARGIN = '300px'

/** Target byte size for splitting markdown into lazy sections. */
export const LAZY_SECTION_TARGET_CHARS = 2500

/** Coefficient and clamps for estimating chunk-block height before render. */
export const CHUNK_HEIGHT_MIN_PX = 100
export const CHUNK_HEIGHT_MAX_PX = 800
export const CHUNK_HEIGHT_PX_PER_CHAR = 0.4

// ── PDF viewer ─────────────────────────────────────────────────────────────

/** Pages to pre-render above and below the visible viewport. */
export const PDF_RENDER_BUFFER = 2

/** Number of pages fetched in parallel when computing dimensions. */
export const PDF_DIM_BATCH = 20

// ── Chunk operations ───────────────────────────────────────────────────────

/** Tail length scanned when detecting overlap between adjacent chunks during
 *  merge. Larger values catch longer overlaps at the cost of more comparisons. */
export const CHUNK_OVERLAP_SCAN_TAIL = 300

// ── Chunk colours ──────────────────────────────────────────────────────────

export const CHUNK_COLORS = [
  'rgba(139, 69, 19, 0.12)', 'rgba(61, 107, 39, 0.12)', 'rgba(204, 34, 0, 0.09)',
  'rgba(180, 140, 60, 0.15)', 'rgba(30, 90, 140, 0.10)', 'rgba(210, 105, 30, 0.13)',
  'rgba(90, 50, 120, 0.09)', 'rgba(20, 110, 100, 0.11)', 'rgba(160, 60, 30, 0.12)',
  'rgba(60, 120, 60, 0.12)',
]

export const CHUNK_BORDER_COLORS = [
  '#8B4513', '#3D6B27', '#CC2200', '#B48C3C', '#1E5A8C',
  '#D2691E', '#5A3278', '#146E64', '#A03C1E', '#3C783C',
]
