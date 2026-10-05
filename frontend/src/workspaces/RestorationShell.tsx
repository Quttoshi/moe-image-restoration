import { useState, type ReactNode } from 'react'
import type { CorruptionSettings, ImageSource, RestoreBase } from '../api'
import { CorruptionControls } from '../components/CorruptionControls'
import { InputPicker } from '../components/InputPicker'
import { ErrorBox, ImagePanel, Section, Spinner, Stat, Timing } from '../components/ui'

interface Props<R extends RestoreBase> {
  title: string
  subtitle: string
  run: (src: ImageSource, s: CorruptionSettings) => Promise<R>
  extraControls?: ReactNode
  renderInsights?: (r: R) => ReactNode
  outputCaption?: (r: R) => ReactNode
}

export function RestorationShell<R extends RestoreBase>({ title, subtitle, run, extraControls, renderInsights, outputCaption }: Props<R>) {
  const [src, setSrc] = useState<ImageSource>({})
  const [preview, setPreview] = useState<string | null>(null)
  const [settings, setSettings] = useState<CorruptionSettings>({ corruption: 'salt_pepper', severity: 'medium' })
  const [result, setResult] = useState<R | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const tag = title.toLowerCase().replace(/[^a-z]+/g, '-')

  const go = async () => {
    setBusy(true)
    setError(null)
    try {
      setResult(await run(src, settings))
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
          <h2 className="text-xl font-bold text-slate-900">{title}</h2>
          <p className="mt-1 text-sm text-slate-500">{subtitle}</p>
        </div>
        <Section title="1 · Input image">
          <InputPicker group="pets" value={src} onChange={(s, p) => { setSrc(s); setPreview(p); setResult(null) }} />
        </Section>
        <Section title="2 · Corruption">
          <CorruptionControls value={settings} onChange={setSettings} />
          {extraControls && <div className="mt-3 border-t border-slate-100 pt-3">{extraControls}</div>}
        </Section>
        <button className="btn-primary w-full py-3" disabled={busy || (!src.file && !src.sample)} onClick={go}>
          {busy ? <Spinner /> : null}
          {busy ? 'Running…' : 'Run restoration'}
        </button>
        <ErrorBox error={error} />
      </aside>

      <main className="space-y-4">
        <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
          <ImagePanel title="Clean reference" src={result?.clean ?? (settings.corruption !== 'none' ? preview : null)} caption={result && !result.clean ? 'Unknown for pre-corrupted uploads' : undefined} />
          <ImagePanel title="Model input" src={result?.input ?? (settings.corruption === 'none' ? preview : null)} caption={result?.corruption.label} download={`${tag}-input.png`} />
          <ImagePanel title="Restored output" src={result?.output} highlight={!!result} caption={result && outputCaption?.(result)} download={`${tag}-output.png`} />
          <ImagePanel title="|Error| map" src={result?.error_map} caption={result?.error_map ? 'vs. clean reference' : undefined} />
        </div>

        {result && (
          <>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Stat label="PSNR input" value={result.psnr_input !== undefined ? `${result.psnr_input} dB` : 'n/a'} />
              <Stat label="PSNR output" value={result.psnr_output !== undefined ? `${result.psnr_output} dB` : 'n/a'} sub={result.psnr_input !== undefined && result.psnr_output !== undefined ? `${(result.psnr_output - result.psnr_input >= 0 ? '+' : '')}${(result.psnr_output - result.psnr_input).toFixed(2)} dB` : undefined} />
              <Stat label="Source" value={<span className="text-sm">{result.source.split(':')[0]}</span>} sub={result.source.split(':').slice(1).join(':')} />
              <Stat label="Corruption" value={<span className="text-sm">{result.corruption.type}</span>} sub={result.corruption.label} />
            </div>
            {renderInsights?.(result)}
            <Section title="Inference time">
              <Timing t={result.timing_ms} />
            </Section>
            <details className="card p-4 text-sm">
              <summary className="cursor-pointer font-semibold text-slate-700">Corruption settings (JSON)</summary>
              <pre className="mt-2 overflow-x-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-100">{JSON.stringify(result.corruption, null, 2)}</pre>
            </details>
          </>
        )}
        {!result && (
          <div className="card flex h-40 items-center justify-center text-sm text-slate-400">Choose an image and corruption, then run the model.</div>
        )}
      </main>
    </div>
  )
}
