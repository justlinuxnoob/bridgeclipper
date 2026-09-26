import { create } from 'zustand'
import { errorMessage } from '../lib/utils'
import { getApi } from '../lib/ipc'
import type { ClipSettings, ToolStatus } from '../../preload/index'

interface SettingsState extends ClipSettings {
  loaded: boolean
  saving: boolean
  toolStatus: ToolStatus | null
  checkingTools: boolean
  toolError: string | null
  load: () => Promise<void>
  save: (settings: Partial<ClipSettings>) => Promise<void>
  replaceApiKey: (key: 'openrouterApiKey' | 'zernioApiKey' | 'groqApiKey', value: string) => Promise<void>
  checkTools: () => Promise<void>
}

// Queue writes so each partial update merges with the last successful save.
let saveQueue: Promise<void> = Promise.resolve()
let pendingSaves = 0
let latestToolCheck = 0

export const useSettingsStore = create<SettingsState>((set, get) => ({
  openrouterConfigured: false,
  zernioConfigured: false,
  groqConfigured: false,
  outputDirectory: '',
  pythonPath: 'python3',
  customVocabulary: '',
  plannerBackend: 'openrouter',
  transcriptionBackend: 'openrouter',
  localWhisperModel: 'large-v3-turbo-q5_0',
  loaded: false,
  saving: false,
  toolStatus: null,
  checkingTools: false,
  toolError: null,

  load: async () => {
    const settings = await getApi().settings.load()
    set({ ...pickSettings(settings), loaded: true })
  },

  save: (updates) => {
    const patch = { ...updates }
    pendingSaves += 1
    set({ saving: true })
    const task = saveQueue.then(async () => {
      const merged: ClipSettings = { ...pickSettings(get()), ...patch }
      const saved = await getApi().settings.save(merged)
      set({ ...pickSettings(saved), loaded: true })
    })
    saveQueue = task.catch(() => {})
    return task.finally(() => {
      pendingSaves -= 1
      set({ saving: pendingSaves > 0 })
    })
  },

  replaceApiKey: (key, value) => {
    pendingSaves += 1
    set({ saving: true })
    const task = saveQueue.then(async () => {
      const saved = await getApi().settings.replaceApiKey(key, value)
      set({ ...pickSettings(saved), loaded: true })
    })
    saveQueue = task.catch(() => {})
    return task.finally(() => {
      pendingSaves -= 1
      set({ saving: pendingSaves > 0 })
    })
  },

  checkTools: async () => {
    const request = ++latestToolCheck
    set({ checkingTools: true, toolError: null })
    try {
      const status = await getApi().system.checkTools()
      if (request === latestToolCheck) set({ toolStatus: status })
    } catch (err) {
      if (request === latestToolCheck) {
        set({ toolStatus: null, toolError: errorMessage(err, 'Could not check required tools. Try again in Settings.') })
      }
    } finally {
      if (request === latestToolCheck) set({ checkingTools: false })
    }
  }
}))

function pickSettings(s: ClipSettings): ClipSettings {
  return {
    openrouterConfigured: s.openrouterConfigured,
    zernioConfigured: s.zernioConfigured,
    groqConfigured: s.groqConfigured,
    outputDirectory: s.outputDirectory,
    pythonPath: s.pythonPath,
    customVocabulary: s.customVocabulary,
    plannerBackend: s.plannerBackend,
    transcriptionBackend: s.transcriptionBackend,
    localWhisperModel: s.localWhisperModel
  }
}

export type ProviderKey = 'OpenRouter' | 'Groq'
export type SetupState = { ready: boolean; missingKeys: ProviderKey[]; toolsOk: boolean | null }

/** Keys the selected AI engines need; mirrors missingProviderKeys in the main process. */
export function missingProviderKeys(s: Pick<ClipSettings, 'plannerBackend' | 'transcriptionBackend' | 'openrouterConfigured' | 'groqConfigured'>): ProviderKey[] {
  const needsOpenRouter = s.plannerBackend === 'openrouter' || s.transcriptionBackend === 'openrouter'
  return [
    needsOpenRouter && !s.openrouterConfigured && 'OpenRouter',
    s.transcriptionBackend === 'groq' && !s.groqConfigured && 'Groq'
  ].filter(Boolean) as ProviderKey[]
}

/** Whether a clip job can start: the keys the selected engines need are present
 *  and, once the system check has run, every required tool found. */
export function useSetupState(): SetupState {
  const openrouterConfigured = useSettingsStore((s) => s.openrouterConfigured)
  const groqConfigured = useSettingsStore((s) => s.groqConfigured)
  const plannerBackend = useSettingsStore((s) => s.plannerBackend)
  const transcriptionBackend = useSettingsStore((s) => s.transcriptionBackend)
  const tools = useSettingsStore((s) => s.toolStatus)
  const toolError = useSettingsStore((s) => s.toolError)
  const checkingTools = useSettingsStore((s) => s.checkingTools)
  const missingKeys = missingProviderKeys({ plannerBackend, transcriptionBackend, openrouterConfigured, groqConfigured })
  const toolsOk = toolError ? false : tools
    ? tools.python && tools.pythonDeps && tools.ffmpeg && tools.ffprobe && tools.ytdlp && tools.engine && tools.bridgeRunner
    : null
  return { ready: missingKeys.length === 0 && toolsOk === true && !checkingTools, missingKeys, toolsOk }
}
