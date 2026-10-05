export type Corruption = 'none' | 'salt_pepper' | 'blur' | 'occlusion'
export type Severity = 'low' | 'medium' | 'high' | 'random' | 'custom'

export interface CorruptionSettings {
  corruption: Corruption
  severity: Severity
  seed?: number
  p?: number
  kernel?: number
  sigma?: number
  n_rects?: number
  area?: number
}

export interface ImageSource {
  file?: File
  sample?: string
}

export interface CorruptionInfo {
  type: string
  label: string
  [k: string]: unknown
}

export interface RestoreBase {
  task: string
  source: string
  clean: string | null
  input: string
  output: string
  corruption: CorruptionInfo
  psnr_input?: number
  psnr_output?: number
  error_map?: string
  timing_ms: Record<string, number>
}

export interface HardResult extends RestoreBase {
  routing_mode: 'predicted' | 'oracle'
  probabilities: Record<string, number>
  predicted: string
  confidence: number
  selected_expert: string
  selected_expert_label: string
  true_label: string | null
  correct: boolean | null
}

export interface SoftResult extends RestoreBase {
  weights: Record<string, number>
  dominant: string
  ranking: string[]
  normalized_entropy: number
  routing_style: 'dominant' | 'distributed'
  true_label: string | null
}

export interface SketchResult {
  task: string
  source: string
  style: number
  style_name: string
  photo: string
  sketch: string
  timing_ms: Record<string, number>
}

export interface Health {
  status: 'ok' | 'degraded'
  uptime_s: number
  loaded: Record<string, { file: string; providers: string[] }>
  missing: Record<string, string>
  onnxruntime: string
  available_providers: string[]
}

function form(src: ImageSource, extra: Record<string, string | number | undefined>): FormData {
  const fd = new FormData()
  if (src.file) fd.append('file', src.file)
  else if (src.sample) fd.append('sample', src.sample)
  for (const [k, v] of Object.entries(extra)) if (v !== undefined && v !== '') fd.append(k, String(v))
  return fd
}

async function post<T>(path: string, fd: FormData): Promise<T> {
  const r = await fetch(`/api${path}`, { method: 'POST', body: fd })
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try {
      const j = await r.json()
      msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
    } catch {
      /* keep status text */
    }
    throw new Error(msg)
  }
  return r.json()
}

const settingsFields = (s: CorruptionSettings) => ({ ...s }) as Record<string, string | number | undefined>

export const api = {
  health: (): Promise<Health> => fetch('/api/health').then((r) => r.json()),
  info: () => fetch('/api/info').then((r) => r.json()),
  samples: (): Promise<{ pets: string[]; faces: string[] }> => fetch('/api/samples').then((r) => r.json()),
  sampleUrl: (name: string) => `/api/samples/${name}`,
  corrupt: (src: ImageSource, s: CorruptionSettings) => post<{ input: string; clean: string; corruption: CorruptionInfo }>('/corrupt', form(src, settingsFields(s))),
  universal: (src: ImageSource, s: CorruptionSettings) => post<RestoreBase>('/restore/universal', form(src, settingsFields(s))),
  hard: (src: ImageSource, s: CorruptionSettings, routing: 'predicted' | 'oracle') =>
    post<HardResult>('/restore/hard', form(src, { ...settingsFields(s), routing })),
  soft: (src: ImageSource, s: CorruptionSettings) => post<SoftResult>('/restore/soft', form(src, settingsFields(s))),
  sketch: (src: ImageSource, style: number) => post<SketchResult>('/sketch', form(src, { style })),
}

export const CLASS_LABEL: Record<string, string> = {
  clean: 'Clean',
  identity: 'Identity',
  salt_pepper: 'Salt & pepper',
  blur: 'Gaussian blur',
  occlusion: 'Occlusion',
}

export const CLASS_COLOR: Record<string, string> = {
  clean: 'var(--color-identity)',
  identity: 'var(--color-identity)',
  salt_pepper: 'var(--color-salt)',
  blur: 'var(--color-blur)',
  occlusion: 'var(--color-occl)',
}
