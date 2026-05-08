import type { MarkdownVersion } from '../types'
import { API_BASE } from './apiService'

/** Response shape of ``GET /api/documents/{name}/markdowns/{identifier}``. */
export interface MarkdownContent {
  filename: string
  source: 'converted' | 'uploaded'
  converter: string | null
  content: string
}

/** List every available Markdown variant for *filename*. */
export async function listMarkdownVersions(filename: string): Promise<MarkdownVersion[]> {
  const res = await fetch(
    `${API_BASE}/documents/${encodeURIComponent(filename)}/markdowns`,
  )
  if (!res.ok) return []
  const data = await res.json().catch(() => ({ versions: [] }))
  return (data.versions ?? []) as MarkdownVersion[]
}

/**
 * Fetch the content of one specific Markdown variant.
 * ``identifier`` may be a converter name (``pymupdf4llm``, ``docling``…) or
 * the raw filename for uploaded MDs.
 */
export async function getMarkdownContent(
  filename: string,
  identifier: string,
): Promise<MarkdownContent> {
  const res = await fetch(
    `${API_BASE}/documents/${encodeURIComponent(filename)}/markdowns/${encodeURIComponent(identifier)}`,
  )
  if (!res.ok) throw new Error(`Get markdown failed: HTTP ${res.status}`)
  return res.json()
}
