import { useEffect, useRef, type KeyboardEvent, type PointerEvent } from 'react'
import { ImagePlus, TriangleAlert } from 'lucide-react'
import { cn } from '../lib/utils'
import { useLogoStore } from '../store/use-logo-store'
import { useSettingsStore } from '../store/use-settings-store'
import type { ClipDraft } from '../store/use-draft-store'
import {
  LOGO_OPACITY_RANGE,
  LOGO_WIDTH_RANGE,
  PLATFORM_UI_ZONES,
  clampLogoPlacement,
  defaultLogoPlacement,
  overlapsPlatformUi,
  type LogoPlacement,
  type LogoPreview
} from '../../shared/logo'
import { Button } from './ui/Button'
import { SettingRow } from './ui/SettingRow'
import { Switch } from './ui/Switch'

interface LogoPlacementEditorProps {
  logo: LogoPreview
  placement: LogoPlacement
  opacity: number
  landscape?: boolean
  disabled?: boolean
  onChange: (placement: LogoPlacement) => void
}

/**
 * The output frame with the logo on top: drag the logo to move it, drag its
 * corner to resize. Arrow keys nudge (Shift for bigger steps), +/- resize.
 * 9:16 frames show where the platforms draw their own buttons and captions.
 */
export function LogoPlacementEditor({ logo, placement, opacity, landscape = false, disabled, onChange }: LogoPlacementEditorProps): React.JSX.Element {
  const frameRef = useRef<HTMLDivElement>(null)
  const drag = useRef<{ mode: 'move' | 'resize'; x: number; y: number; start: LogoPlacement; width: number; height: number } | null>(null)
  const logoAspect = logo.height / logo.width
  const frameAspect = landscape ? 16 / 9 : 9 / 16
  const clamp = (next: LogoPlacement): LogoPlacement => clampLogoPlacement(next, logoAspect, frameAspect)

  const begin = (mode: 'move' | 'resize') => (e: PointerEvent<HTMLElement>): void => {
    const frame = frameRef.current
    if (disabled || !frame || e.button !== 0) return
    e.preventDefault()
    e.stopPropagation()
    const rect = frame.getBoundingClientRect()
    drag.current = { mode, x: e.clientX, y: e.clientY, start: placement, width: rect.width, height: rect.height }
    e.currentTarget.setPointerCapture(e.pointerId)
  }
  const move = (e: PointerEvent<HTMLElement>): void => {
    const d = drag.current
    if (!d) return
    const dx = (e.clientX - d.x) / d.width
    const dy = (e.clientY - d.y) / d.height
    onChange(clamp(d.mode === 'move'
      ? { ...d.start, x: d.start.x + dx, y: d.start.y + dy }
      : { ...d.start, width: d.start.width + dx }))
  }
  const end = (): void => { drag.current = null }

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>): void => {
    if (disabled) return
    const step = e.shiftKey ? 0.05 : 0.01
    const moves: Record<string, Partial<LogoPlacement>> = {
      ArrowLeft: { x: placement.x - step },
      ArrowRight: { x: placement.x + step },
      ArrowUp: { y: placement.y - step },
      ArrowDown: { y: placement.y + step },
      '+': { width: placement.width + 0.01 },
      '=': { width: placement.width + 0.01 },
      '-': { width: placement.width - 0.01 }
    }
    const patch = moves[e.key]
    if (!patch) return
    e.preventDefault()
    onChange(clamp({ ...placement, ...patch }))
  }

  return (
    <div
      ref={frameRef}
      className={cn(
        'relative shrink-0 overflow-hidden rounded-lg bg-black/40 shadow-[inset_0_0_0_1px_rgb(255_255_255/0.1)]',
        landscape ? 'w-72' : 'w-40'
      )}
      style={{ aspectRatio: landscape ? '16 / 9' : '9 / 16' }}
      data-testid="logo-frame"
    >
      {!landscape && PLATFORM_UI_ZONES.map((zone, index) => (
        <div
          key={index}
          aria-hidden
          className="pointer-events-none absolute border border-dashed border-white/25 bg-white/[0.03]"
          style={{ left: `${zone.x * 100}%`, top: `${zone.y * 100}%`, width: `${zone.width * 100}%`, height: `${zone.height * 100}%` }}
        />
      ))}
      <div
        role="group"
        tabIndex={disabled ? -1 : 0}
        aria-label="Logo position. Drag, or use the arrow keys to move and + or - to resize."
        className="absolute cursor-move touch-none select-none rounded-sm outline-none ring-accent/80 hover:ring-1 focus-visible:ring-2"
        style={{ left: `${placement.x * 100}%`, top: `${placement.y * 100}%`, width: `${placement.width * 100}%` }}
        onPointerDown={begin('move')}
        onPointerMove={move}
        onPointerUp={end}
        onPointerCancel={end}
        onKeyDown={onKeyDown}
      >
        <img src={logo.dataUrl} alt="" draggable={false} className="block h-auto w-full" style={{ opacity }} />
        <span
          aria-hidden
          className="absolute -bottom-1.5 -right-1.5 h-3 w-3 cursor-nwse-resize rounded-sm bg-white shadow-[0_0_0_1px_rgb(0_0_0/0.45)]"
          onPointerDown={begin('resize')}
          onPointerMove={move}
          onPointerUp={end}
          onPointerCancel={end}
        />
      </div>
    </div>
  )
}

/** The Create form's logo controls: on/off, pick or remove the saved logo, and place it. */
export function LogoSection({ draft, update }: { draft: ClipDraft; update: (patch: Partial<ClipDraft>) => void }): React.JSX.Element {
  const logoPath = useSettingsStore((s) => s.logoPath)
  const { preview, loading, error, load, select, remove } = useLogoStore()
  useEffect(() => { void load() }, [logoPath, load])

  const landscape = draft.aspectRatio === '16:9'
  const logoAspect = preview ? preview.height / preview.width : 1
  const frameAspect = landscape ? 16 / 9 : 9 / 16
  const placement = clampLogoPlacement(draft.logoPlacement, logoAspect, frameAspect)
  const enabled = draft.logoEnabled && Boolean(preview)
  const overlap = enabled && !landscape && overlapsPlatformUi(placement, logoAspect)
  const setPlacement = (logoPlacement: LogoPlacement): void => update({ logoPlacement })

  const choose = async (): Promise<void> => {
    const chosen = await select()
    if (chosen) update({ logoEnabled: true, logoPlacement: defaultLogoPlacement(chosen.height / chosen.width, frameAspect) })
  }

  return (
    <div className="space-y-3">
      <SettingRow
        title="Logo watermark"
        description={preview ? 'Burned into every clip, above captions and the title.' : 'Add a transparent PNG to brand every clip.'}
        control={<Switch label="Logo watermark" checked={enabled} disabled={!preview} onChange={(logoEnabled) => update({ logoEnabled })} />}
      />
      {error && <p role="alert" className="text-xs text-danger">{error}</p>}
      {preview ? (
        <div className="flex flex-wrap items-start gap-4">
          <div className={cn('transition-[opacity,filter] duration-300 ease-out', !enabled && 'pointer-events-none opacity-35 saturate-50')} aria-disabled={!enabled}>
            <LogoPlacementEditor logo={preview} placement={placement} opacity={draft.logoOpacity} landscape={landscape} disabled={!enabled} onChange={setPlacement} />
          </div>
          <div className="min-w-[12rem] flex-1 space-y-3">
            <div className={cn('space-y-3 transition-[opacity,filter] duration-300 ease-out', !enabled && 'pointer-events-none opacity-35 saturate-50')} aria-disabled={!enabled}>
              <RangeRow
                label="Size"
                value={placement.width}
                min={LOGO_WIDTH_RANGE[0]}
                max={LOGO_WIDTH_RANGE[1]}
                disabled={!enabled}
                onChange={(width) => setPlacement(clampLogoPlacement({ ...placement, width }, logoAspect, frameAspect))}
              />
              <RangeRow
                label="Opacity"
                value={draft.logoOpacity}
                min={LOGO_OPACITY_RANGE[0]}
                max={LOGO_OPACITY_RANGE[1]}
                disabled={!enabled}
                onChange={(logoOpacity) => update({ logoOpacity })}
              />
              {overlap && (
                <p role="status" className="flex items-start gap-1.5 text-2xs leading-relaxed text-warning">
                  <TriangleAlert className="mt-px h-3.5 w-3.5 shrink-0" />
                  The logo overlaps where TikTok, Shorts and Reels draw their buttons and captions, so it may be covered.
                </p>
              )}
              <p className="text-2xs leading-relaxed text-ink-subtle">
                Drag to move, drag the corner to resize.{!landscape && ' Dashed areas are covered by platform buttons.'} This logo is saved as your default.
              </p>
            </div>
            {/* Replace and remove stay available while the logo is switched off for this job. */}
            <div className="flex flex-wrap gap-1">
              <Button size="sm" variant="ghost" disabled={!enabled} onClick={() => setPlacement(defaultLogoPlacement(logoAspect, frameAspect))}>Reset position</Button>
              <Button size="sm" variant="ghost" loading={loading} onClick={() => void choose()}>Replace…</Button>
              <Button size="sm" variant="ghost" onClick={() => void remove()}>Remove logo</Button>
            </div>
          </div>
        </div>
      ) : (
        <Button variant="secondary" size="sm" icon={<ImagePlus className="h-3.5 w-3.5" />} loading={loading} onClick={() => void choose()}>
          Choose logo PNG…
        </Button>
      )}
    </div>
  )
}

function RangeRow({ label, value, min, max, disabled, onChange }: {
  label: string
  value: number
  min: number
  max: number
  disabled?: boolean
  onChange: (value: number) => void
}): React.JSX.Element {
  return (
    <label className="flex items-center gap-3 text-xs text-ink-muted">
      <span className="w-14 shrink-0">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={0.01}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="min-w-0 flex-1 accent-[rgb(var(--accent))]"
      />
      <span className="w-9 shrink-0 text-right tabular-nums text-ink-subtle">{Math.round(value * 100)}%</span>
    </label>
  )
}
