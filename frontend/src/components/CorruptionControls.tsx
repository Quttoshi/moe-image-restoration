import type { Corruption, CorruptionSettings, Severity } from '../api'

const TYPES: { id: Corruption; label: string }[] = [
  { id: 'none', label: 'None (use input as-is)' },
  { id: 'salt_pepper', label: 'Salt & pepper' },
  { id: 'blur', label: 'Gaussian blur' },
  { id: 'occlusion', label: 'Occlusion' },
]

const SEV: { id: Severity; label: string }[] = [
  { id: 'low', label: 'Low' },
  { id: 'medium', label: 'Medium' },
  { id: 'high', label: 'High' },
  { id: 'random', label: 'Random' },
  { id: 'custom', label: 'Custom' },
]

const LEVEL_HINT: Record<string, Record<string, string>> = {
  salt_pepper: { low: 'p = 0.03', medium: 'p = 0.08', high: 'p = 0.15', random: 'p ~ U(0.02, 0.15)' },
  blur: { low: 'k = 3, σ = 0.7', medium: 'k = 5, σ = 1.5', high: 'k = 7, σ = 2.5', random: 'k ∈ {3,5,7}, σ ~ U(0.5, 2.5)' },
  occlusion: { low: '1 rect, ~10%', medium: '2 rects, ~20%', high: '3 rects, ~35%', random: '1–3 rects, 10–35%' },
}

export function CorruptionControls({ value, onChange }: { value: CorruptionSettings; onChange: (s: CorruptionSettings) => void }) {
  const set = (patch: Partial<CorruptionSettings>) => onChange({ ...value, ...patch })
  const num = (v: string) => (v === '' ? undefined : Number(v))
  const field = 'w-full rounded-lg border border-slate-200 px-2 py-1.5 text-sm focus:border-brand-500 focus:outline-none'

  return (
    <div className="space-y-3">
      <div>
        <p className="label mb-1.5">Runtime corruption</p>
        <div className="grid grid-cols-2 gap-1.5">
          {TYPES.map((t) => (
            <button
              key={t.id}
              onClick={() => set({ corruption: t.id })}
              className={`rounded-lg border px-2 py-1.5 text-left text-sm transition ${value.corruption === t.id ? 'border-brand-600 bg-brand-50 font-semibold text-brand-700' : 'border-slate-200 hover:border-slate-300'}`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {value.corruption !== 'none' && (
        <>
          <div>
            <p className="label mb-1.5">Severity</p>
            <div className="flex flex-wrap gap-1.5">
              {SEV.map((s) => (
                <button
                  key={s.id}
                  onClick={() => set({ severity: s.id })}
                  className={`rounded-full border px-3 py-1 text-xs font-medium transition ${value.severity === s.id ? 'border-brand-600 bg-brand-600 text-white' : 'border-slate-200 text-slate-600 hover:border-slate-300'}`}
                >
                  {s.label}
                </button>
              ))}
            </div>
            {value.severity !== 'custom' && <p className="mt-1.5 font-mono text-xs text-slate-500">{LEVEL_HINT[value.corruption][value.severity]}</p>}
          </div>

          {value.severity === 'custom' && (
            <div className="grid grid-cols-2 gap-2">
              {value.corruption === 'salt_pepper' && (
                <label className="col-span-2 text-xs text-slate-600">
                  Probability p
                  <input className={field} type="number" step="0.01" min="0.01" max="0.5" value={value.p ?? ''} onChange={(e) => set({ p: num(e.target.value) })} />
                </label>
              )}
              {value.corruption === 'blur' && (
                <>
                  <label className="text-xs text-slate-600">
                    Kernel
                    <select className={field} value={value.kernel ?? ''} onChange={(e) => set({ kernel: num(e.target.value) })}>
                      <option value="">–</option>
                      {[3, 5, 7, 9].map((k) => (
                        <option key={k}>{k}</option>
                      ))}
                    </select>
                  </label>
                  <label className="text-xs text-slate-600">
                    Sigma
                    <input className={field} type="number" step="0.1" min="0.1" max="5" value={value.sigma ?? ''} onChange={(e) => set({ sigma: num(e.target.value) })} />
                  </label>
                </>
              )}
              {value.corruption === 'occlusion' && (
                <>
                  <label className="text-xs text-slate-600">
                    Rectangles
                    <select className={field} value={value.n_rects ?? ''} onChange={(e) => set({ n_rects: num(e.target.value) })}>
                      <option value="">–</option>
                      {[1, 2, 3].map((k) => (
                        <option key={k}>{k}</option>
                      ))}
                    </select>
                  </label>
                  <label className="text-xs text-slate-600">
                    Area (0–0.5)
                    <input className={field} type="number" step="0.01" min="0.02" max="0.5" value={value.area ?? ''} onChange={(e) => set({ area: num(e.target.value) })} />
                  </label>
                </>
              )}
            </div>
          )}

          <label className="block text-xs text-slate-600">
            Seed (optional, for reproducible masks/noise)
            <input className={field} type="number" value={value.seed ?? ''} onChange={(e) => set({ seed: num(e.target.value) })} />
          </label>
        </>
      )}
    </div>
  )
}
