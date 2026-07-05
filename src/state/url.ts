import type { ModelArchitecture } from '../engine/types'
import { MODEL_PRESETS } from '../data/models'

/**
 * Calculator state serialized into the URL so configurations are
 * shareable. Custom (HF-imported) models are embedded as JSON.
 */
export interface CalculatorState {
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
}

export const DEFAULT_STATE: CalculatorState = {
  modelName: 'Llama 3.1 8B',
  weightFormat: 'bf16',
  kvFormat: 'fp16',
  contextLength: 8192,
  concurrentSequences: 8,
  tensorParallel: 1,
  pipelineParallel: 1,
  gpuId: 'h100-sxm',
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
  p.set('w', state.weightFormat)
  p.set('kv', state.kvFormat)
  p.set('ctx', String(state.contextLength))
  p.set('seqs', String(state.concurrentSequences))
  p.set('tp', String(state.tensorParallel))
  p.set('pp', String(state.pipelineParallel))
  p.set('gpu', state.gpuId)
  return p
}

export function stateFromParams(p: URLSearchParams): CalculatorState {
  const s: CalculatorState = { ...DEFAULT_STATE }
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
  return s
}
