import { app } from 'electron'
import { randomUUID } from 'crypto'
import { closeSync, constants, fstatSync, lstatSync, mkdirSync, openSync, readdirSync, readSync, rmSync, writeFileSync } from 'fs'
import { basename, dirname, isAbsolute, join, resolve } from 'path'
import { LOGO_MAX_BYTES, LOGO_MAX_DIMENSION, type LogoPreview } from '../shared/logo'

/**
 * The logo watermark. A picked PNG is checked for transparency and copied into
 * userData/logos, so the path the job carries never points at a file the user
 * can move or edit later, and the engine only ever reads files we wrote.
 */

export interface PngInfo {
  width: number
  height: number
  /** IHDR colour type 4 (grey + alpha) or 6 (RGBA), or a tRNS chunk. */
  transparent: boolean
}

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
const LOGO_NAME = /^logo-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.png$/

/** Reads the PNG header and walks the chunks before image data. Throws for anything that isn't a PNG. */
export function inspectPng(bytes: Uint8Array): PngInfo {
  const notPng = new Error('Choose a PNG image.')
  if (bytes.length < 33 || PNG_SIGNATURE.some((byte, i) => bytes[i] !== byte)) throw notPng
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const chunkType = (offset: number): string => String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3])
  // IHDR must come first and is always 13 bytes.
  if (view.getUint32(8) !== 13 || chunkType(12) !== 'IHDR') throw notPng
  const width = view.getUint32(16)
  const height = view.getUint32(20)
  const colorType = bytes[25]
  if (width === 0 || height === 0 || ![0, 2, 3, 4, 6].includes(colorType)) throw notPng
  let transparent = colorType === 4 || colorType === 6
  // tRNS, when present, must come before the first IDAT.
  let offset = 33
  while (!transparent && offset + 8 <= bytes.length) {
    const length = view.getUint32(offset)
    const type = chunkType(offset + 4)
    if (type === 'IDAT' || type === 'IEND') break
    if (type === 'tRNS') transparent = true
    offset += 12 + length
  }
  return { width, height, transparent }
}

/** Reads and checks a logo candidate: a regular file, at most 5 MB, a PNG with transparency. */
export function readLogoFile(path: string): { bytes: Buffer; info: PngInfo } {
  if (typeof path !== 'string' || !isAbsolute(path) || path.includes('\0')) throw new Error('Choose a PNG image.')
  let fd: number
  try {
    fd = openSync(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0))
  } catch {
    throw new Error('Could not read that logo file.')
  }
  try {
    const stat = fstatSync(fd)
    if (!stat.isFile()) throw new Error('Choose a PNG image.')
    if (stat.size > LOGO_MAX_BYTES) throw new Error('The logo must be 5 MB or smaller.')
    const bytes = Buffer.alloc(stat.size)
    let read = 0
    while (read < bytes.length) {
      const n = readSync(fd, bytes, read, bytes.length - read, read)
      if (n === 0) break
      read += n
    }
    const info = inspectPng(bytes.subarray(0, read))
    if (info.width > LOGO_MAX_DIMENSION || info.height > LOGO_MAX_DIMENSION) {
      throw new Error(`The logo must be at most ${LOGO_MAX_DIMENSION}×${LOGO_MAX_DIMENSION} pixels.`)
    }
    if (!info.transparent) {
      throw new Error('This PNG has no transparency. Export the logo with a transparent background (PNG with alpha) and try again.')
    }
    return { bytes: bytes.subarray(0, read), info }
  } finally {
    closeSync(fd)
  }
}

export function logoDirectory(): string {
  return join(app.getPath('userData'), 'logos')
}

/** A logo copy this app wrote: directly inside userData/logos, with our generated name, a regular file. */
export function isManagedLogoPath(path: unknown): path is string {
  if (typeof path !== 'string' || !isAbsolute(path) || path.includes('\0') || resolve(path) !== path) return false
  if (dirname(path) !== logoDirectory() || !LOGO_NAME.test(basename(path))) return false
  try {
    const stat = lstatSync(path)
    return stat.isFile() && stat.size <= LOGO_MAX_BYTES
  } catch { return false }
}

/** Throws unless `path` is a managed logo copy that still passes the PNG checks. */
export function assertManagedLogo(path: unknown): asserts path is string {
  if (!isManagedLogoPath(path)) throw new Error('Choose the logo again from the Create form.')
  readLogoFile(path)
}

/** Copies a picked PNG into userData/logos and returns the managed copy. */
export function importLogo(source: string): LogoPreview {
  const { bytes, info } = readLogoFile(source)
  const directory = logoDirectory()
  mkdirSync(directory, { recursive: true, mode: 0o700 })
  const target = join(directory, `logo-${randomUUID()}.png`)
  writeFileSync(target, bytes, { flag: 'wx', mode: 0o600 })
  return { path: target, width: info.width, height: info.height, dataUrl: toDataUrl(bytes) }
}

/** The saved logo for the Create form, or null when it is missing or no longer valid. */
export function loadLogoPreview(path: string): LogoPreview | null {
  if (!path || !isManagedLogoPath(path)) return null
  try {
    const { bytes, info } = readLogoFile(path)
    return { path, width: info.width, height: info.height, dataUrl: toDataUrl(bytes) }
  } catch { return null }
}

/** Deletes managed logo copies except `keep` (the saved logo and any still used by queued or running jobs). */
export function pruneLogos(keep: Iterable<string | null | undefined>): void {
  const keepSet = new Set([...keep].filter(Boolean))
  let names: string[]
  try { names = readdirSync(logoDirectory()) } catch { return }
  for (const name of names) {
    const path = join(logoDirectory(), name)
    if (LOGO_NAME.test(name) && !keepSet.has(path)) {
      try { rmSync(path, { force: true }) } catch { /* Best effort; a later prune retries. */ }
    }
  }
}

function toDataUrl(bytes: Buffer): string {
  return `data:image/png;base64,${bytes.toString('base64')}`
}
