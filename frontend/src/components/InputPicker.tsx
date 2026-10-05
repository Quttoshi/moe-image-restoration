import { useEffect, useRef, useState } from 'react'
import { api, type ImageSource } from '../api'

interface Props {
  group: 'pets' | 'faces'
  value: ImageSource
  onChange: (src: ImageSource, preview: string | null) => void
  webcam?: boolean
}

/** Upload / sample gallery / optional webcam capture. */
export function InputPicker({ group, value, onChange, webcam = false }: Props) {
  const [samples, setSamples] = useState<string[]>([])
  const [mode, setMode] = useState<'upload' | 'sample' | 'webcam'>('sample')
  const [dragOver, setDragOver] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    api.samples().then((s) => setSamples(s[group] ?? [])).catch(() => setSamples([]))
  }, [group])

  const pickFile = (f: File | undefined) => {
    if (!f) return
    onChange({ file: f }, URL.createObjectURL(f))
  }

  const tabs: { id: typeof mode; label: string }[] = [
    { id: 'sample', label: 'Sample' },
    { id: 'upload', label: 'Upload' },
    ...(webcam ? [{ id: 'webcam' as const, label: 'Webcam' }] : []),
  ]

  return (
    <div className="space-y-3">
      <div className="flex gap-1 rounded-xl bg-slate-100 p-1">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setMode(t.id)}
            className={`flex-1 rounded-lg px-3 py-1.5 text-sm font-medium transition ${mode === t.id ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {mode === 'upload' && (
        <div
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            pickFile(e.dataTransfer.files[0])
          }}
          onClick={() => fileRef.current?.click()}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 py-8 text-center transition ${dragOver ? 'border-brand-500 bg-brand-50' : 'border-slate-300 hover:border-brand-500'}`}
        >
          <span className="text-sm font-medium text-slate-700">{value.file ? value.file.name : 'Drop an image or click to browse'}</span>
          <span className="mt-1 text-xs text-slate-400">PNG, JPEG, WebP or BMP, max 10 MB. Resized to 128×128.</span>
          <input ref={fileRef} type="file" accept="image/png,image/jpeg,image/webp,image/bmp" hidden onChange={(e) => pickFile(e.target.files?.[0])} />
        </div>
      )}

      {mode === 'sample' && (
        <div className="grid max-h-56 grid-cols-4 gap-2 overflow-y-auto pr-1">
          {samples.length === 0 && <p className="col-span-4 text-sm text-slate-400">No samples available.</p>}
          {samples.map((s) => (
            <button
              key={s}
              onClick={() => onChange({ sample: s }, api.sampleUrl(s))}
              className={`overflow-hidden rounded-lg border-2 transition ${value.sample === s ? 'border-brand-600 ring-2 ring-brand-100' : 'border-transparent hover:border-slate-300'}`}
            >
              <img src={api.sampleUrl(s)} alt={s} className="aspect-square w-full object-cover" />
            </button>
          ))}
        </div>
      )}

      {mode === 'webcam' && <Webcam onCapture={(f, url) => onChange({ file: f }, url)} />}
    </div>
  )
}

function Webcam({ onCapture }: { onCapture: (f: File, url: string) => void }) {
  const video = useRef<HTMLVideoElement>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let stream: MediaStream | null = null
    navigator.mediaDevices
      ?.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } })
      .then((s) => {
        stream = s
        if (video.current) video.current.srcObject = s
      })
      .catch((e) => setError(e instanceof Error ? e.message : 'Camera unavailable'))
    if (!navigator.mediaDevices) setError('Camera API unavailable (requires HTTPS or localhost)')
    return () => stream?.getTracks().forEach((t) => t.stop())
  }, [])

  const capture = () => {
    const v = video.current
    if (!v || !v.videoWidth) return
    const side = Math.min(v.videoWidth, v.videoHeight)
    const c = document.createElement('canvas')
    c.width = c.height = side
    const ctx = c.getContext('2d')!
    // Mirror horizontally so the capture matches the preview the user sees.
    ctx.translate(side, 0)
    ctx.scale(-1, 1)
    ctx.drawImage(v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side, 0, 0, side, side)
    c.toBlob((b) => b && onCapture(new File([b], 'webcam.png', { type: 'image/png' }), c.toDataURL('image/png')), 'image/png')
  }

  if (error) return <p className="rounded-xl bg-rose-50 p-3 text-sm text-rose-700">{error}</p>
  return (
    <div className="space-y-2">
      <div className="relative overflow-hidden rounded-xl bg-slate-900">
        <video ref={video} autoPlay playsInline muted className="aspect-square w-full -scale-x-100 object-cover" />
        <div className="pointer-events-none absolute inset-[15%] rounded-full border-2 border-dashed border-white/60" />
      </div>
      <button className="btn-ghost w-full" onClick={capture}>
        Capture photo
      </button>
    </div>
  )
}
