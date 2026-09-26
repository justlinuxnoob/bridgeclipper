import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { Check, Copy, Instagram, Music2, X, Youtube, type LucideIcon } from 'lucide-react'
import type { PostCaptions, SocialPostText } from '../../shared/job-output'
import { getApi } from '../lib/ipc'
import { Button } from './ui/Button'
import { Dialog, DialogFooter } from './ui/Dialog'

interface Props {
  title: string
  captions: PostCaptions
  onClose: () => void
}

function socialText(block: SocialPostText): string {
  return [block.caption, block.hashtags.join(' ')].filter(Boolean).join('\n\n')
}

/** Ready-to-post text for TikTok, YouTube Shorts and Instagram Reels, one copy button per field. */
export function PostCaptionsDialog({ title, captions, onClose }: Props): React.JSX.Element {
  const titleId = useId()
  const dialogRef = useRef<HTMLDivElement>(null)
  const close = useCallback(() => onClose(), [onClose])

  useEffect(() => {
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const dialog = dialogRef.current
    dialog?.focus()
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') { event.preventDefault(); close(); return }
      if (event.key !== 'Tab' || !dialog) return
      const focusable = [...dialog.querySelectorAll<HTMLElement>('button:not([disabled])')]
      if (!focusable.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) { event.preventDefault(); last.focus() }
      else if (!event.shiftKey && (document.activeElement === last || !dialog.contains(document.activeElement))) { event.preventDefault(); first.focus() }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => { document.removeEventListener('keydown', onKeyDown); previousFocus?.focus() }
  }, [close])

  const { tiktok, youtube, instagram } = captions
  return (
    <Dialog ref={dialogRef} aria-labelledby={titleId} onBackdropMouseDown={close} panelClassName="max-w-[560px]">
      <div className="flex items-start justify-between gap-3 px-5 pt-5">
        <div className="min-w-0">
          <h2 id={titleId} className="text-lg font-semibold text-ink">Post captions</h2>
          <p className="mt-1 truncate text-sm text-ink-muted">{title}</p>
        </div>
        <Button variant="ghost" iconOnly aria-label="Close" icon={<X className="h-4 w-4" />} onClick={close} />
      </div>
      <div className="space-y-3 overflow-y-auto px-5 py-5">
        <PlatformBlock icon={Music2} name="TikTok" fields={[{ label: 'Caption', text: socialText(tiktok) }]} />
        <PlatformBlock
          icon={Youtube}
          name="YouTube Shorts"
          fields={[
            { label: 'Title', text: youtube.title },
            { label: 'Description', text: youtube.description },
            { label: 'Tags', text: youtube.tags.join(', ') }
          ]}
        />
        <PlatformBlock icon={Instagram} name="Instagram Reels" fields={[{ label: 'Caption', text: socialText(instagram) }]} />
      </div>
      <DialogFooter>
        <p className="mr-auto text-2xs text-ink-subtle">Written from the clip transcript. Check it before posting.</p>
        <Button onClick={close}>Done</Button>
      </DialogFooter>
    </Dialog>
  )
}

function PlatformBlock({ icon: Icon, name, fields }: { icon: LucideIcon; name: string; fields: { label: string; text: string }[] }): React.JSX.Element {
  return (
    <section className="glass-tile rounded-xl p-3">
      <h3 className="flex items-center gap-2 text-sm font-semibold text-ink"><Icon className="h-4 w-4" />{name}</h3>
      <div className="mt-2 space-y-2">
        {fields.filter((field) => field.text).map((field) => (
          <div key={field.label}>
            <div className="flex items-center justify-between gap-2">
              <span className="eyebrow">{field.label}</span>
              <CopyButton label={`Copy ${name} ${field.label.toLowerCase()}`} text={field.text} />
            </div>
            <p className="mt-1 whitespace-pre-wrap text-sm leading-relaxed text-ink-muted">{field.text}</p>
          </div>
        ))}
      </div>
    </section>
  )
}

function CopyButton({ label, text }: { label: string; text: string }): React.JSX.Element {
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    if (!copied) return
    const timer = setTimeout(() => setCopied(false), 1500)
    return () => clearTimeout(timer)
  }, [copied])
  return (
    <Button
      size="sm"
      variant="ghost"
      aria-label={label}
      icon={copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      onClick={() => { void getApi().shell.copyText(text).then(() => setCopied(true)).catch(() => {}) }}
    >
      {copied ? 'Copied' : 'Copy'}
    </Button>
  )
}
