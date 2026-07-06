import type { ModelArchitecture } from '../engine/types'
import type {
  ActivationCheckpointing,
  FinetuneMethod,
  LoraTargets,
  TrainingOptimizer,
  TrainingPrecision,
  ZeroStage,
} from '../engine/training'
import { MODEL_PRESETS } from '../data/models'

export type Scenario = 'inference' | 'training' | 'llmd'

/**
 * Calculator state serialized into the URL so configurations are
 * shareable. Custom (HF-imported) models are embedded as JSON.
 *
 * `contextLength` does double duty: context length per sequence in
 * inference, sequence length in training — switching scenarios keeps it.
 * Training-only fields are serialized only when scenario === 'training',
 * so inference URLs are unchanged from previous releases.
 */
export interface CalculatorState {
  scenario: Scenario
  /** Preset name, or 'custom' when the model came from HF import. */
  modelName: string
  customModel?: ModelArchitecture
  weightFormat: string
  kvFormat: string
  contextLength: number
  concurrentSequences: number
  tensorParallel: number
  pipelineParallel: number
  gpuId: string
  // --- training scenario ---
  optimizer: TrainingOptimizer
  trainPrecision: TrainingPrecision
  zeroStage: ZeroStage
  checkpointing: ActivationCheckpointing
  /** Data-parallel world size (total GPUs) for training. */
  dataParallel: number
  /** Sequences per GPU per step. */
  microBatch: number
  /** Fine-tuning mode on the training scenario; 'full' is classic training. */
  ftMethod: FinetuneMethod
  loraRank: number
  loraTargets: LoraTargets
  // --- llm-d scenario ---
  /** Mean arrival rate λ, requests per second. */
  requestRate: number
  inputTokens: number
  outputTokens: number
  ttftSloMs: number
  tpotSloMs: number
  prefillGpuId: string
  prefillTp: number
  decodeGpuId: string
  decodeTp: number
  /** Calibratable knobs (docs/interop.md) — serialized only when non-default. */
  llmdMfu: number
  llmdBwEff: number
  llmdKvBeta: number
}

export const DEFAULT_STATE: CalculatorState = {
  scenario: 'inference',
  modelName: 'Llama 3.1 8B',
  weightFormat: 'bf16',
  kvFormat: 'fp16',
  contextLength: 8192,
  concurrentSequences: 8,
  tensorParallel: 1,
  pipelineParallel: 1,
  gpuId: 'h100-sxm',
  optimizer: 'adamw',
  trainPrecision: 'mixed-bf16',
  zeroStage: 2,
  checkpointing: 'selective',
  dataParallel: 8,
  microBatch: 1,
  ftMethod: 'full',
  loraRank: 16,
  loraTargets: 'attn-qv',
  requestRate: 50,
  inputTokens: 1024,
  outputTokens: 256,
  ttftSloMs: 500,
  tpotSloMs: 100,
  prefillGpuId: 'h100-sxm',
  prefillTp: 1,
  decodeGpuId: 'h100-sxm',
  decodeTp: 1,
  llmdMfu: 0.45,
  llmdBwEff: 0.7,
  llmdKvBeta: 1.8,
}

export function resolveModel(state: CalculatorState): ModelArchitecture {
  if (state.modelName === 'custom' && state.customModel) return state.customModel
  return MODEL_PRESETS.find((m) => m.name === state.modelName) ?? MODEL_PRESETS[2]
}

export function stateToParams(state: CalculatorState): URLSearchParams {
  const p = new URLSearchParams()
  p.set('model', state.modelName)
  if (state.modelName === 'custom' && state.customModel) {
    p.set('arch', btoa(JSON.stringify(state.customModel)))
  }
  if (state.scenario === 'training') {
    p.set('sc', 'training')
    p.set('tprec', state.trainPrecision)
    p.set('opt', state.optimizer)
    p.set('zero', String(state.zeroStage))
    p.set('ckpt', state.checkpointing)
    p.set('ctx', String(state.contextLength))
    p.set('mb', String(state.microBatch))
    p.set('dp', String(state.dataParallel))
    if (state.ftMethod !== 'full') {
      p.set('ft', state.ftMethod)
      p.set('rank', String(state.loraRank))
      p.set('targ', state.loraTargets)
    }
  } else if (state.scenario === 'llmd') {
    p.set('sc', 'llmd')
    p.set('w', state.weightFormat)
    p.set('kv', state.kvFormat)
    p.set('rate', String(state.requestRate))
    p.set('in', String(state.inputTokens))
    p.set('out', String(state.outputTokens))
    p.set('ttft', String(state.ttftSloMs))
    p.set('tpot', String(state.tpotSloMs))
    p.set('pgpu', state.prefillGpuId)
    p.set('ptp', String(state.prefillTp))
    p.set('dgpu', state.decodeGpuId)
    p.set('dtp', String(state.decodeTp))
    if (state.llmdMfu !== DEFAULT_STATE.llmdMfu) p.set('mfu', String(state.llmdMfu))
    if (state.llmdBwEff !== DEFAULT_STATE.llmdBwEff) p.set('beff', String(state.llmdBwEff))
    if (state.llmdKvBeta !== DEFAULT_STATE.llmdKvBeta) p.set('beta', String(state.llmdKvBeta))
  } else {
    p.set('w', state.weightFormat)
    p.set('kv', state.kvFormat)
    p.set('ctx', String(state.contextLength))
    p.set('seqs', String(state.concurrentSequences))
    p.set('tp', String(state.tensorParallel))
    p.set('pp', String(state.pipelineParallel))
  }
  p.set('gpu', state.gpuId)
  return p
}

function oneOf<T extends string>(v: string | null, allowed: readonly T[], fallback: T): T {
  return allowed.includes(v as T) ? (v as T) : fallback
}

export function stateFromParams(p: URLSearchParams): CalculatorState {
  const s: CalculatorState = { ...DEFAULT_STATE }
  s.scenario = oneOf(p.get('sc'), ['inference', 'training', 'llmd'] as const, 'inference')
  const model = p.get('model')
  if (model) s.modelName = model
  const arch = p.get('arch')
  if (arch) {
    try {
      s.customModel = JSON.parse(atob(arch)) as ModelArchitecture
    } catch {
      s.modelName = DEFAULT_STATE.modelName
    }
  }
  const num = (key: string, fallback: number, min = 1) => {
    const v = Number(p.get(key))
    return Number.isFinite(v) && v >= min ? v : fallback
  }
  s.weightFormat = p.get('w') ?? s.weightFormat
  s.kvFormat = p.get('kv') ?? s.kvFormat
  s.contextLength = num('ctx', s.contextLength)
  s.concurrentSequences = num('seqs', s.concurrentSequences)
  s.tensorParallel = num('tp', s.tensorParallel)
  s.pipelineParallel = num('pp', s.pipelineParallel)
  s.gpuId = p.get('gpu') ?? s.gpuId
  s.optimizer = oneOf(p.get('opt'), ['adamw', 'adamw-8bit', 'sgd'] as const, s.optimizer)
  s.trainPrecision = oneOf(p.get('tprec'), ['mixed-bf16', 'fp32'] as const, s.trainPrecision)
  const zeroRaw = p.get('zero')
  if (zeroRaw !== null && [0, 1, 2, 3].includes(Number(zeroRaw))) s.zeroStage = Number(zeroRaw) as ZeroStage
  s.checkpointing = oneOf(p.get('ckpt'), ['none', 'selective', 'full'] as const, s.checkpointing)
  s.dataParallel = num('dp', s.dataParallel)
  s.microBatch = num('mb', s.microBatch)
  s.ftMethod = oneOf(p.get('ft'), ['full', 'lora', 'qlora'] as const, s.ftMethod)
  s.loraRank = num('rank', s.loraRank)
  s.loraTargets = oneOf(p.get('targ'), ['attn-qv', 'attn-all', 'all-linear'] as const, s.loraTargets)
  // llm-d params. A knob outside its plausible range falls back to the
  // documented default rather than silently producing nonsense estimates.
  const knob = (key: string, fallback: number, min: number, max: number) => {
    const raw = p.get(key)
    if (raw === null) return fallback
    const v = Number(raw)
    return Number.isFinite(v) && v >= min && v <= max ? v : fallback
  }
  s.requestRate = knob('rate', s.requestRate, 0, 1e6)
  s.inputTokens = num('in', s.inputTokens)
  s.outputTokens = num('out', s.outputTokens)
  s.ttftSloMs = num('ttft', s.ttftSloMs)
  s.tpotSloMs = num('tpot', s.tpotSloMs)
  s.prefillGpuId = p.get('pgpu') ?? s.prefillGpuId
  s.prefillTp = num('ptp', s.prefillTp)
  s.decodeGpuId = p.get('dgpu') ?? s.decodeGpuId
  s.decodeTp = num('dtp', s.decodeTp)
  s.llmdMfu = knob('mfu', s.llmdMfu, 0.05, 1)
  s.llmdBwEff = knob('beff', s.llmdBwEff, 0.05, 1)
  s.llmdKvBeta = knob('beta', s.llmdKvBeta, 1, 5)
  return s
}
