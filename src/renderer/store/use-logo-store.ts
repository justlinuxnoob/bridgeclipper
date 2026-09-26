import { create } from 'zustand'
import { errorMessage } from '../lib/utils'
import { getApi } from '../lib/ipc'
import type { LogoPreview } from '../../shared/logo'
import { useSettingsStore } from './use-settings-store'

interface LogoState {
  /** The saved logo, loaded once per path. */
  preview: LogoPreview | null
  loading: boolean
  error: string | null
  /** Loads the preview for Settings' saved logo if it isn't loaded yet. */
  load: () => Promise<void>
  /** Opens the PNG picker; the chosen logo becomes the saved default. */
  select: () => Promise<LogoPreview | null>
  remove: () => Promise<void>
}

let latestLoad = 0

export const useLogoStore = create<LogoState>((set, get) => ({
  preview: null,
  loading: false,
  error: null,

  load: async () => {
    const path = useSettingsStore.getState().logoPath
    if (!path) { set({ preview: null }); return }
    if (get().preview?.path === path) return
    const request = ++latestLoad
    set({ loading: true })
    try {
      const preview = await getApi().logo.load()
      if (request !== latestLoad) return
      set({ preview, error: preview ? null : 'The saved logo is missing or no longer valid. Choose it again.' })
    } catch {
      if (request === latestLoad) set({ preview: null })
    } finally {
      if (request === latestLoad) set({ loading: false })
    }
  },

  select: async () => {
    latestLoad++
    set({ loading: true, error: null })
    try {
      const preview = await getApi().logo.select()
      if (preview) {
        set({ preview })
        useSettingsStore.setState({ logoPath: preview.path })
      }
      return preview
    } catch (err) {
      set({ error: errorMessage(err, 'Could not use that logo.') })
      return null
    } finally {
      set({ loading: false })
    }
  },

  remove: async () => {
    set({ error: null })
    try {
      const saved = await getApi().logo.remove()
      useSettingsStore.setState({ logoPath: saved.logoPath })
      set({ preview: null })
    } catch (err) {
      set({ error: errorMessage(err, 'Could not remove the logo.') })
    }
  }
}))

