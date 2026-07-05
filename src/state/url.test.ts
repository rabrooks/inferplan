import { describe, expect, it } from 'vitest'
import { DEFAULT_STATE, stateFromParams, stateToParams, type CalculatorState } from './url'

describe('URL round-trip', () => {
  it('inference state round-trips and stays free of training params', () => {
    const state: CalculatorState = {
      ...DEFAULT_STATE,
      modelName: 'Llama 3.3 70B',
      weightFormat: 'int4',
      kvFormat: 'fp8',
      contextLength: 32768,
      concurrentSequences: 16,
      tensorParallel: 4,
      pipelineParallel: 2,
      gpuId: 'a100-80',
    }
    const p = stateToParams(state)
    expect(p.get('sc')).toBeNull()
    expect(p.get('zero')).toBeNull()
    expect(stateFromParams(p)).toEqual(state)
  })

  it('training state round-trips through the URL', () => {
    const state: CalculatorState = {
      ...DEFAULT_STATE,
      scenario: 'training',
      modelName: 'Llama 3.1 8B',
      contextLength: 4096,
      microBatch: 4,
      dataParallel: 32,
      zeroStage: 3,
      optimizer: 'adamw-8bit',
      trainPrecision: 'mixed-bf16',
      checkpointing: 'full',
      gpuId: 'h200',
    }
    const parsed = stateFromParams(stateToParams(state))
    expect(parsed.scenario).toBe('training')
    expect(parsed.zeroStage).toBe(3)
    expect(parsed.optimizer).toBe('adamw-8bit')
    expect(parsed.checkpointing).toBe('full')
    expect(parsed.dataParallel).toBe(32)
    expect(parsed.microBatch).toBe(4)
    expect(parsed.contextLength).toBe(4096)
    expect(parsed.gpuId).toBe('h200')
  })

  it('a pre-training-release inference URL still parses (defaults fill the new fields)', () => {
    const p = new URLSearchParams('model=Llama+3.1+8B&w=bf16&kv=fp16&ctx=8192&seqs=8&tp=1&pp=1&gpu=h100-sxm')
    const s = stateFromParams(p)
    expect(s.scenario).toBe('inference')
    expect(s.zeroStage).toBe(DEFAULT_STATE.zeroStage)
    expect(s.dataParallel).toBe(DEFAULT_STATE.dataParallel)
  })

  it('rejects garbage enum values instead of propagating them', () => {
    const p = new URLSearchParams('sc=training&opt=evil&tprec=fp64&zero=9&ckpt=maybe')
    const s = stateFromParams(p)
    expect(s.optimizer).toBe(DEFAULT_STATE.optimizer)
    expect(s.trainPrecision).toBe(DEFAULT_STATE.trainPrecision)
    expect(s.zeroStage).toBe(DEFAULT_STATE.zeroStage)
    expect(s.checkpointing).toBe(DEFAULT_STATE.checkpointing)
  })
})
