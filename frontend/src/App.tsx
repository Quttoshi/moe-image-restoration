import { useEffect, useState } from 'react'
import { api, type Health } from './api'
import { SketchWorkspace } from './workspaces/SketchWorkspace'
import { HardRoutedWorkspace, SoftMoEWorkspace, UniversalWorkspace } from './workspaces/Workspaces'

const TABS = [
  { id: 'overview', label: 'Overview', short: 'Home' },
  { id: 'universal', label: 'Universal Restoration', short: 'Task 1' },
  { id: 'hard', label: 'Hard-Routed Restoration', short: 'Task 2' },
  { id: 'soft', label: 'Soft Mixture-of-Experts Restoration', short: 'Task 3' },
  { id: 'sketch', label: 'Face-to-Sketch Generator', short: 'Task 4' },
] as const
type TabId = (typeof TABS)[number]['id']

const currentTab = (): TabId => {
  const h = window.location.hash.slice(1)
  return (TABS.find((t) => t.id === h)?.id ?? 'overview') as TabId
}

export default function App() {
  const [tab, setTab] = useState<TabId>(currentTab)
  const [health, setHealth] = useState<Health | null>(null)
  const [healthErr, setHealthErr] = useState(false)

  useEffect(() => {
    const onHash = () => setTab(currentTab())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    const poll = () =>
      api
        .health()
        .then((h) => {
          setHealth(h)
          setHealthErr(false)
        })
        .catch(() => setHealthErr(true))
    poll()
    const id = setInterval(poll, 15000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3">
          <a href="#overview" className="flex items-center gap-2">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-brand-600 font-bold text-white">G</span>
            <span>
              <span className="block text-sm font-bold leading-tight text-slate-900">GenAI Restoration Studio</span>
              <span className="block text-xs text-slate-500">21i2525 · Assignment 1</span>
            </span>
          </a>
          <nav className="flex flex-1 gap-1 overflow-x-auto">
            {TABS.map((t) => (
              <a key={t.id} href={`#${t.id}`} className={`whitespace-nowrap rounded-lg px-3 py-2 text-sm font-medium transition ${tab === t.id ? 'bg-brand-50 text-brand-700' : 'text-slate-600 hover:bg-slate-100'}`}>
                <span className="hidden xl:inline">{t.label}</span>
                <span className="xl:hidden">{t.short}</span>
              </a>
            ))}
          </nav>
          <HealthBadge health={health} error={healthErr} />
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-4 py-6">
        {tab === 'overview' && <Overview health={health} />}
        {tab === 'universal' && <UniversalWorkspace />}
        {tab === 'hard' && <HardRoutedWorkspace />}
        {tab === 'soft' && <SoftMoEWorkspace />}
        {tab === 'sketch' && <SketchWorkspace />}
      </div>
    </div>
  )
}

function HealthBadge({ health, error }: { health: Health | null; error: boolean }) {
  const state = error ? 'offline' : !health ? 'connecting' : health.status
  const color = { ok: 'bg-emerald-500', degraded: 'bg-amber-500', offline: 'bg-rose-500', connecting: 'bg-slate-400' }[state]
  return (
    <a href="#overview" className="flex items-center gap-2 rounded-full border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600" title="Backend health">
      <span className={`h-2 w-2 rounded-full ${color}`} />
      API {state}
    </a>
  )
}

const CARDS = [
  { id: 'universal', task: 'Task 1', title: 'Universal Restoration', body: 'A single convolutional autoencoder with a genuine bottleneck restores salt-and-pepper noise, blur and occlusion without being told the corruption type.' },
  { id: 'hard', task: 'Task 2', title: 'Hard-Routed Restoration', body: 'A 4-way corruption classifier sends each image to exactly one specialist autoencoder. Clean images skip restoration.' },
  { id: 'soft', task: 'Task 3', title: 'Soft Mixture-of-Experts', body: 'A gate initialised from the classifier blends the identity branch and three experts with continuous weights. Gate and experts are fine-tuned jointly.' },
  { id: 'sketch', task: 'Task 4', title: 'Face-to-Sketch Generator', body: 'A style-conditioned pix2pix cGAN: U-Net generator, PatchGAN discriminator and a learned style embedding over three FS2K sketch styles.' },
]

function Overview({ health }: { health: Health | null }) {
  return (
    <div className="space-y-6">
      <section className="card bg-gradient-to-br from-brand-600 to-indigo-800 p-8 text-white">
        <p className="text-sm font-semibold uppercase tracking-wider text-indigo-200">Generative AI · Assignment 1 · 21i2525</p>
        <h1 className="mt-2 text-3xl font-bold">Image restoration and face-to-sketch generation</h1>
        <p className="mt-2 max-w-2xl text-indigo-100">
          Four generative systems trained in PyTorch, tuned with Optuna, tracked in MLflow, exported to ONNX and served by FastAPI with ONNX Runtime. Pick a workspace to upload an image, apply a corruption at runtime and inspect the model's output.
        </p>
      </section>
      <div className="grid gap-4 md:grid-cols-2">
        {CARDS.map((c) => (
          <a key={c.id} href={`#${c.id}`} className="card group p-5 transition hover:border-brand-500 hover:shadow-md">
            <p className="label text-brand-600">{c.task}</p>
            <h3 className="mt-1 text-lg font-bold text-slate-900 group-hover:text-brand-700">{c.title}</h3>
            <p className="mt-1 text-sm text-slate-500">{c.body}</p>
          </a>
        ))}
      </div>
      <SystemPanel health={health} />
    </div>
  )
}

function SystemPanel({ health }: { health: Health | null }) {
  if (!health) return <div className="card p-5 text-sm text-slate-400">Waiting for backend…</div>
  return (
    <section className="card p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-semibold text-slate-900">System information</h3>
        <span className="font-mono text-xs text-slate-500">
          onnxruntime {health.onnxruntime} · uptime {Math.round(health.uptime_s)} s · providers {health.available_providers.join(', ')}
        </span>
      </div>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="label border-b border-slate-100">
              <th className="py-2">Model</th>
              <th>ONNX file</th>
              <th>Execution provider</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(health.loaded).map(([k, m]) => (
              <tr key={k} className="border-b border-slate-50">
                <td className="py-2 font-medium">{k}</td>
                <td className="font-mono text-xs">{m.file}</td>
                <td className="font-mono text-xs">{m.providers[0]}</td>
                <td><span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-semibold text-emerald-700">loaded</span></td>
              </tr>
            ))}
            {Object.entries(health.missing).map(([k, why]) => (
              <tr key={k} className="border-b border-slate-50">
                <td className="py-2 font-medium">{k}</td>
                <td className="font-mono text-xs" colSpan={2}>{why}</td>
                <td><span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs font-semibold text-amber-700">missing</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
