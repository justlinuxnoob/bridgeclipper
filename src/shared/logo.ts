/**
 * Logo watermark placement, shared by the Create form and the main process.
 *
 * Placement is normalized to the output frame: `x`/`y` are the logo's top-left
 * corner as fractions of the frame width/height, `width` is a fraction of the
 * frame width, and the height follows from the logo's own aspect ratio.
 */
export interface LogoPlacement {
  x: number
  y: number
  width: number
}

/** The logo sent with a job: the managed copy under userData plus its placement. */
export interface LogoOverlay extends LogoPlacement {
  path: string
  /** 0–1, multiplied into the PNG's own alpha. Defaults to 1. */
  opacity?: number
}

/** The saved logo as the Create form previews it. */
export interface LogoPreview {
  path: string
  width: number
  height: number
  /** data:image/png;base64,… (the renderer's CSP allows data: images). */
  dataUrl: string
}

export const LOGO_MAX_BYTES = 5 * 1024 * 1024
export const LOGO_MAX_DIMENSION = 8192
export const LOGO_WIDTH_RANGE = [0.05, 0.5] as const
export const LOGO_OPACITY_RANGE = [0.2, 1] as const
export const LOGO_MARGIN = 0.04
export const DEFAULT_LOGO_WIDTH = 0.15

/** Top-right corner, inset by the margin on both axes (margin measured in frame widths). */
export function defaultLogoPlacement(logoAspect: number, frameAspect = 9 / 16): LogoPlacement {
  return clampLogoPlacement({ x: 1 - LOGO_MARGIN - DEFAULT_LOGO_WIDTH, y: LOGO_MARGIN * frameAspect, width: DEFAULT_LOGO_WIDTH }, logoAspect, frameAspect)
}

/**
 * Logo height as a fraction of the frame height.
 * `logoAspect` is logo height / logo width; `frameAspect` is frame width / frame height.
 */
export function logoHeightFraction(width: number, logoAspect: number, frameAspect = 9 / 16): number {
  return width * logoAspect * frameAspect
}

/** Keep the width in range and the whole logo inside the frame. */
export function clampLogoPlacement(placement: LogoPlacement, logoAspect: number, frameAspect = 9 / 16): LogoPlacement {
  const aspect = Number.isFinite(logoAspect) && logoAspect > 0 ? logoAspect : 1
  let width = clamp(placement.width, LOGO_WIDTH_RANGE[0], LOGO_WIDTH_RANGE[1])
  // A tall logo on a wide frame could otherwise overflow vertically.
  const maxWidthForHeight = 1 / (aspect * frameAspect)
  if (width > maxWidthForHeight) width = maxWidthForHeight
  const height = logoHeightFraction(width, aspect, frameAspect)
  return {
    x: clamp(placement.x, 0, 1 - width),
    y: clamp(placement.y, 0, Math.max(0, 1 - height)),
    width
  }
}

/** Where TikTok, Shorts and Reels draw their own buttons and captions on a 9:16 frame. */
export const PLATFORM_UI_ZONES: readonly { x: number; y: number; width: number; height: number }[] = [
  // Like / comment / share column on the right edge.
  { x: 0.84, y: 0.35, width: 0.16, height: 0.45 },
  // Username, caption and sound ticker along the bottom.
  { x: 0, y: 0.8, width: 1, height: 0.2 }
]

/** Whether a 9:16 placement overlaps the platform UI guide (a soft warning only). */
export function overlapsPlatformUi(placement: LogoPlacement, logoAspect: number): boolean {
  const height = logoHeightFraction(placement.width, logoAspect)
  return PLATFORM_UI_ZONES.some((zone) =>
    placement.x < zone.x + zone.width && placement.x + placement.width > zone.x &&
    placement.y < zone.y + zone.height && placement.y + height > zone.y)
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}
