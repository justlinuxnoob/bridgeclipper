/**
 * What kind of video a job clips. Podcast plans from the transcript only;
 * Streamer and Gambling also send loud audio moments (hype) to the planner.
 */
export const CONTENT_MODES = [
  { id: 'podcast', label: 'Podcast', hint: 'Talk, interviews: planned from the transcript' },
  { id: 'streamer', label: 'Streamer', hint: 'Streams, gaming: also finds loud reactions' },
  { id: 'gambling', label: 'Gambling', hint: 'Casino streams: also finds big-win reactions' }
] as const

export type ContentMode = (typeof CONTENT_MODES)[number]['id']

export function isContentMode(value: unknown): value is ContentMode {
  return CONTENT_MODES.some((mode) => mode.id === value)
}
