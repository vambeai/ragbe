import type { Capabilities, EnrichmentSettings } from '../types'
import { parseSse } from '../utils/parseSse'
import { DEFAULT_ENRICHMENT_BASE_URL, DEFAULT_ENRICHMENT_TEMPERATURE } from '../hooks/useSettings'

export const API_BASE = '/api'

/**
 * Map a wire-level converter name (the value used by the /api/convert
 * endpoint) to the lowercase token the backend uses for filenames and
 * dropdown identifiers.  Kept in sync with
 * ``backend.services.document_service._CONVERTER_NORMALIZATION``; only the
 * identity-different entries need to appear here.
 */
const CONVERTER_FILENAME_TOKEN: Record<string, string> = {
  pymupdf: 'pymupdf4llm',
}

export function converterFilenameToken(converter: string | undefined | null): string | null {
  if (!converter) return null
  return CONVERTER_FILENAME_TOKEN[converter] ?? converter
}

/**
 * Set of MD-source tokens that the conversion pipeline produces.  Mirrors
 * ``backend.utils.naming.KNOWN_MD_SOURCES`` minus ``"uploaded"``: that one
 * is the fallback, not a token that ever appears as the suffix in a
 * converted filename.
 */
const KNOWN_CONVERTER_TOKENS = new Set([
  'pymupdf4llm', 'docling', 'markitdown', 'liteparse', 'vlm', 'cloud',
])

/**
 * Derive the MD-source token (e.g. ``"docling"``, ``"uploaded"``) from a
 * Markdown filename without depending on ``availableMarkdowns`` being
 * populated.  Used by the chunks auto-link logic so the source is known
 * the moment ``documentData.md_filename`` updates, not only after the
 * version listing has loaded.
 */
export function mdSourceFromFilename(
  mdFilename: string | null | undefined,
  pdfFilename: string | null | undefined,
): string | null {
  if (!mdFilename) return null
  const mdStem = mdFilename.replace(/\.md$/i, '')
  const docStem = (pdfFilename ?? '').replace(/\.pdf$/i, '')
  if (docStem) {
    const prefix = `${docStem}_`
    if (mdStem.startsWith(prefix)) {
      const token = mdStem.slice(prefix.length)
      if (KNOWN_CONVERTER_TOKENS.has(token)) return token
    }
  }
  return 'uploaded'
}

/**
 * Build the request body for enrichment endpoints.
 * Applies defaults for optional fields so callers don't need to repeat them.
 */
export function buildEnrichmentBody(
  settings: EnrichmentSettings,
  extra: Record<string, unknown>,
): Record<string, unknown> {
  return {
    ...extra,
    settings: {
      model: settings.model,
      base_url: settings.base_url ?? DEFAULT_ENRICHMENT_BASE_URL,
      api_key: settings.api_key ?? 'ollama',
      temperature: settings.temperature ?? DEFAULT_ENRICHMENT_TEMPERATURE,
      user_prompt: settings.user_prompt,
    },
  }
}

export const capabilityService = {
  async get(): Promise<Capabilities> {
    const res = await fetch(`${API_BASE}/capabilities`)
    if (!res.ok) throw new Error('Failed to fetch capabilities')
    return res.json()
  },
}

/**
 * Enrich a single markdown section via SSE.
 * Returns the enriched content string.
 * Throws on error; throws DOMException(AbortError) if cancelled/aborted.
 */
export async function apiEnrichMarkdown(
  settings: EnrichmentSettings,
  content: string,
  signal?: AbortSignal,
  onConnectionLost?: () => void,
): Promise<string> {
  const res = await fetch(`${API_BASE}/enrich/markdown`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify(buildEnrichmentBody(settings, { content })),
  })
  if (!res.ok) {
    const errText = await res.text().catch(() => res.statusText)
    throw new Error(`Enrichment failed ${res.status}: ${errText}`)
  }
  if (!res.body) throw new Error('No response body')
  for await (const event of parseSse(res.body, onConnectionLost)) {
    if (event.type === 'done') return event.enriched_content as string
    if (event.type === 'error') throw new Error(String(event.message ?? 'Enrichment error'))
    if (event.type === 'cancelled') throw new DOMException('Enrichment cancelled', 'AbortError')
  }
  throw new Error('Stream ended without a done event')
}

/**
 * Enrich a single chunk via SSE (sends a one-item batch).
 * Returns the enriched chunk fields as a plain object.
 * Throws on error; throws DOMException(AbortError) if cancelled/aborted.
 */
export async function apiEnrichChunk(
  settings: EnrichmentSettings,
  index: number,
  content: string,
  start: number,
  end: number,
  metadata: Record<string, unknown>,
  signal?: AbortSignal,
  onConnectionLost?: () => void,
): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/enrich/chunks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify(buildEnrichmentBody(settings, {
      chunks: [{ index, content, start, end, metadata }],
    })),
  })
  if (!res.ok) {
    const errText = await res.text().catch(() => res.statusText)
    throw new Error(`Chunk enrichment failed ${res.status}: ${errText}`)
  }
  if (!res.body) throw new Error('No response body')
  for await (const event of parseSse(res.body, onConnectionLost)) {
    if (event.type === 'chunk_done') return event.chunk as Record<string, unknown>
    if (event.type === 'error') throw new Error(String(event.message ?? 'Chunk enrichment error'))
    if (event.type === 'cancelled') throw new DOMException('Chunk enrichment cancelled', 'AbortError')
  }
  throw new Error('Stream ended without a done event')
}
