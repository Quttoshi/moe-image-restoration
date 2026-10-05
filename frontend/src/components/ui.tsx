import type { ReactNode } from 'react'
import { CLASS_COLOR, CLASS_LABEL } from '../api'

export function ImagePanel({ title, src, caption, download, highlight }: { title: string; src?: string | null; caption?: ReactNode; download?: string; highlight?: boolean }) {
  return (
    <figure className={`card overflow-hidden ${highlight ? 'ring-2 ring-brand-500' : ''}`}>
      <div className="flex items-center justify-between border-b border-slate-100 px-3 py-2">
        <span className="label">{title}</span>
        {src && download && (
          <a href={src} download={download} className="text-xs font-semibold text-brand-600 hover:underline">
            Download
          </a>
        )}
      </div>
      <div className="flex aspect-square items-center justify-center bg-[repeating-conic-gradient(#f1f5f9_0_25%,#fff_0_50%)] bg-[length:16px_16px]">
        {src ? <img src={src} alt={title} className="pixelated h-full w-full object-contain" /> : <span className="text-xs text-slate-400">—</span>}
      </div>
      {caption && <figcaption className="px-3 py-2 text-xs text-slate-500">{caption}</figcaption>}
    </figure>
  )
}

/** Horizontal bar list for class probabilities / routing weights. */
export function Bars({ values, highlight, order }: { values: Record<string, number>; highlight?: string; order?: string[] }) {
  const keys = order ?? Object.keys(values)
  return (
    <div className="space-y-2.5">
      {keys.map((k) => {
        const v = values[k] ?? 0
        const on = k === highlight
        return (
          <div key={k}>
            <div className="mb-1 flex justify-between text-sm">
              <span className={on ? 'font-semibold text-slate-900' : 'text-slate-600'}>
                {CLASS_LABEL[k] ?? k}
                {on && <span className="ml-2 rounded-full bg-brand-100 px-2 py-0.5 text-[10px] font-bold uppercase text-brand-700">top</span>}
              </span>
              <span className="font-mono text-slate-700">{(v * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
              <div className="h-full rounded-full transition-all duration-500" style={{ width: `${Math.max(v * 100, 0.5)}%`, background: CLASS_COLOR[k] ?? 'var(--color-brand-500)', opacity: on ? 1 : 0.55 }} />
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="rounded-xl bg-slate-50 px-3 py-2">
      <p className="label">{label}</p>
      <p className="mt-0.5 font-mono text-lg font-semibold text-slate-900">{value}</p>
      {sub && <p className="text-xs text-slate-500">{sub}</p>}
    </div>
  )
}

export function Timing({ t }: { t: Record<string, number> }) {
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      {Object.entries(t).map(([k, v]) => (
        <Stat key={k} label={k} value={`${v.toFixed(1)} ms`} />
      ))}
    </div>
  )
}

export function ErrorBox({ error }: { error: string | null }) {
  if (!error) return null
  return <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</div>
}

export function Section({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="card p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
        {aside}
      </div>
      {children}
    </section>
  )
}

export function Spinner() {
  return <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
}
