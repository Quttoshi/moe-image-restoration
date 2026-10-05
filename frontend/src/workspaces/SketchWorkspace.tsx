import { useState } from 'react'
import { api, type ImageSource, type SketchResult } from '../api'
import { InputPicker } from '../components/InputPicker'
import { ErrorBox, ImagePanel, Section, Spinner, Stat, Timing } from '../components/ui'

const STYLES = [1, 2, 3]

export function SketchWorkspace() {
  const [src, setSrc] = useState<ImageSource>({})
  const [preview, setPreview] = useState<string | null>(null)
  const [style, setStyle] = useState(1)
  const [result, setResult] = useState<SketchResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const go = async () => {
    setBusy(true)
    setError(null)
    try {
      setResult(await api.sketch(src, style))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
      <aside className="space-y-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900">Face-to-Sketch Generator</h2>
          <p className="mt-1 text-sm text-slate-500">Task 4 · A style-conditioned pix2pix (U-Net generator with a PatchGAN discriminator) trained on FS2K. The style is a learned embedding.</p>
        </div>
        <Section title="1 · Facial photograph">
          <InputPicker group="faces" webcam value={src} onChange={(s, p) => { setSrc(s); setPreview(p); setResult(null) }} />
          <p className="mt-2 text-xs text-slate-400">The photo is centre-cropped to a square and resized to 128×128.</p>
        </Section>
        <Section title="2 · Sketch style">
          <div className="grid grid-cols-3 gap-2">
            {STYLES.map((s) => (
              <button key={s} onClick={() => setStyle(s)} className={`rounded-xl border-2 px-2 py-3 text-sm font-semibold transition ${style === s ? 'border-brand-600 bg-brand-50 text-brand-700' : 'border-slate-200 text-slate-600 hover:border-slate-300'}`}>
                Style {s}
              </button>
            ))}
          </div>
        </Section>
        <button className="btn-primary w-full py-3" disabled={busy || (!src.file && !src.sample)} onClick={go}>
          {busy ? <Spinner /> : null}
          {busy ? 'Generating…' : 'Generate sketch'}
        </button>
        <ErrorBox error={error} />
      </aside>

      <main className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2">
          <ImagePanel title="Original photograph" src={result?.photo ?? preview} />
          <ImagePanel title={`Generated sketch${result ? ` · ${result.style_name}` : ''}`} src={result?.sketch} highlight={!!result} download={result ? `sketch-style${result.style}.png` : undefined} />
        </div>
        {result ? (
          <>
            <div className="flex flex-wrap items-center gap-3">
              <a className="btn-primary" href={result.sketch} download={`sketch-style${result.style}.png`}>
                Download sketch
              </a>
              <Stat label="Style condition" value={result.style_name} sub="embedding index fed to G and D" />
              <Stat label="Model" value={<span className="text-sm">t4_generator.onnx</span>} />
            </div>
            <Section title="Inference time">
              <Timing t={result.timing_ms} />
            </Section>
          </>
        ) : (
          <div className="card flex h-40 items-center justify-center text-sm text-slate-400">Upload, pick or capture a face, choose a style and generate.</div>
        )}
      </main>
    </div>
  )
}
