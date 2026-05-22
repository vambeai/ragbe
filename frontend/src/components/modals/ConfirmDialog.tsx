import type { ReactNode } from 'react'

/**
 * A single action button rendered in the dialog's footer.
 *
 * ``label`` accepts ReactNode so callers can swap to a loading
 * indicator (e.g. ``"Saving…"`` while a request is in flight) without
 * recreating the whole dialog.
 */
export interface ConfirmDialogAction {
  label: ReactNode
  onClick: () => void
  variant?: 'primary' | 'secondary' | 'danger'
  disabled?: boolean
}

interface Props {
  isOpen: boolean
  /** Called when the user clicks the dimmed backdrop (acts as Cancel). */
  onDismiss: () => void
  /** Dialog body — typically a single ``<p>`` with the question. */
  children: ReactNode
  /** Footer buttons, left-to-right.  Most dialogs have 2 (cancel + confirm). */
  actions: ConfirmDialogAction[]
}

/**
 * Shared modal-confirmation primitive.
 *
 * Replaces two near-identical overlays in App.tsx that both built the
 * same fixed scaffolding (full-viewport dimmed overlay → centred card
 * with click-outside-to-cancel → message body → action row).  Reuses
 * the existing ``.confirm-switch-*`` classes from App.css so the
 * visual treatment matches what was already shipped.
 *
 * Returns ``null`` when ``isOpen`` is false so callers can render it
 * unconditionally without an extra wrapper conditional.
 */
export default function ConfirmDialog({ isOpen, onDismiss, children, actions }: Props) {
  if (!isOpen) return null
  return (
    <div className="confirm-switch-overlay" onClick={onDismiss}>
      <div className="confirm-switch-dialog" onClick={e => e.stopPropagation()}>
        {children}
        <div className="confirm-switch-actions">
          {actions.map((a, i) => (
            <button
              key={i}
              className={`btn-${a.variant ?? 'secondary'}`}
              onClick={a.onClick}
              disabled={a.disabled}
            >
              {a.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
