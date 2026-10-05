import { useState } from 'react'
import { api, CLASS_LABEL, type HardResult, type SoftResult } from '../api'
import { Bars, Section, Stat } from '../components/ui'
import { RestorationShell } from './RestorationShell'

export function UniversalWorkspace() {
  return (
    <RestorationShell
      title="Universal Restoration"
      subtitle="Task 1 · One convolutional denoising autoencoder with a compressed bottleneck restores every corruption. The model is never told which corruption was applied."
      run={api.universal}
      outputCaption={() => 't1_universal_dae.onnx'}
    />
  )
}

export function HardRoutedWorkspace() {
  const [routing, setRouting] = useState<'predicted' | 'oracle'>('predicted')
  return (
    <RestorationShell<HardResult>
      title="Hard-Routed Restoration"
      subtitle="Task 2 · A CNN classifier predicts the corruption, then exactly one specialist autoencoder restores it. Clean inputs use an identity bypass."
      run={(s, c) => api.hard(s, c, routing)}
      extraControls={
        <div>
          <p className="label mb-1.5">Routing mode</p>
          <div className="flex gap-1 rounded-xl bg-slate-100 p-1">
            {(['predicted', 'oracle'] as const).map((m) => (
              <button key={m} onClick={() => setRouting(m)} className={`flex-1 rounded-lg px-3 py-1.5 text-sm font-medium capitalize ${routing === m ? 'bg-white shadow-sm' : 'text-slate-500'}`}>
                {m}
              </button>
            ))}
          </div>
          <p className="mt-1.5 text-xs text-slate-500">Oracle uses the known corruption label to pick the expert, so it needs a server-applied corruption.</p>
        </div>
      }
      outputCaption={(r) => r.selected_expert_label}
      renderInsights={(r) => (
        <div className="grid gap-4 md:grid-cols-[1fr_280px]">
          <Section title="Classifier probabilities">
            <Bars values={r.probabilities} highlight={r.predicted} order={['clean', 'salt_pepper', 'blur', 'occlusion']} />
          </Section>
          <Section title="Routing decision">
            <div className="space-y-2">
              <Stat label="Predicted corruption" value={<span className="text-base">{CLASS_LABEL[r.predicted]}</span>} sub={`confidence ${(r.confidence * 100).toFixed(1)}%`} />
              <Stat label="Selected expert" value={<span className="text-base">{r.selected_expert_label}</span>} sub={`${r.routing_mode} routing`} />
              {r.true_label && (
                <div className={`rounded-xl px-3 py-2 text-sm font-medium ${r.correct ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'}`}>
                  {r.correct ? '✓ Classifier correct' : `✗ Misrouted: true label is ${CLASS_LABEL[r.true_label]}`}
                </div>
              )}
            </div>
          </Section>
        </div>
      )}
    />
  )
}

const BRANCHES = ['identity', 'salt_pepper', 'blur', 'occlusion']

export function SoftMoEWorkspace() {
  return (
    <RestorationShell<SoftResult>
      title="Soft Mixture-of-Experts Restoration"
      subtitle="Task 3 · A gating network assigns a continuous weight to the identity branch and each of the three experts. The output is their weighted sum."
      run={api.soft}
      outputCaption={(r) => `Σ wₖ·Aₖ(x̃) · dominant: ${CLASS_LABEL[r.dominant]}`}
      renderInsights={(r) => (
        <div className="grid gap-4 md:grid-cols-[1fr_280px]">
          <Section title="Routing weights w = softmax(G(x̃)/τ)">
            <Bars values={r.weights} highlight={r.dominant} order={BRANCHES} />
            <div className="mt-4 flex h-8 overflow-hidden rounded-lg" title="Stacked contribution">
              {BRANCHES.map((b) => (
                <div
                  key={b}
                  className="flex items-center justify-center text-[10px] font-bold text-white"
                  style={{ width: `${r.weights[b] * 100}%`, background: `var(--color-${b === 'identity' ? 'identity' : b === 'salt_pepper' ? 'salt' : b === 'blur' ? 'blur' : 'occl'})` }}
                >
                  {r.weights[b] > 0.08 ? CLASS_LABEL[b] : ''}
                </div>
              ))}
            </div>
          </Section>
          <Section title="Gate behaviour">
            <div className="space-y-2">
              <Stat label="Strongest contributor" value={<span className="text-base">{CLASS_LABEL[r.dominant]}</span>} sub={`${(r.weights[r.dominant] * 100).toFixed(1)}% of the output`} />
              <Stat label="Routing" value={<span className="text-base capitalize">{r.routing_style}</span>} sub={`normalised entropy ${r.normalized_entropy.toFixed(3)}`} />
              <Stat label="Contribution ranking" value={<span className="text-sm">{r.ranking.map((b) => CLASS_LABEL[b]).join(' › ')}</span>} />
            </div>
          </Section>
        </div>
      )}
    />
  )
}
