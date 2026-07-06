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

  it('fine-tuning state round-trips; full FT keeps the URL free of ft params', () => {
    const state: CalculatorState = {
      ...DEFAULT_STATE,
      scenario: 'training',
      ftMethod: 'qlora',
      loraRank: 64,
      loraTargets: 'all-linear',
      optimizer: 'adamw-8bit',
    }
    const p = stateToParams(state)
    expect(p.get('ft')).toBe('qlora')
    const parsed = stateFromParams(p)
    expect(parsed.ftMethod).toBe('qlora')
    expect(parsed.loraRank).toBe(64)
    expect(parsed.loraTargets).toBe('all-linear')
    const fullFt = stateToParams({ ...state, ftMethod: 'full' })
    expect(fullFt.get('ft')).toBeNull()
    expect(fullFt.get('rank')).toBeNull()
    expect(stateFromParams(fullFt).ftMethod).toBe('full')
  })

  it('llm-d state round-trips; default knobs stay out of the URL', () => {
    const state: CalculatorState = {
      ...DEFAULT_STATE,
      scenario: 'llmd',
      modelName: 'DeepSeek-V3 / R1 671B (MoE, MLA)',
      weightFormat: 'fp8',
      kvFormat: 'fp8',
      requestRate: 20,
      inputTokens: 2048,
      outputTokens: 512,
      ttftSloMs: 400,
      tpotSloMs: 50,
      prefillGpuId: 'h200',
      prefillTp: 8,
      decodeGpuId: 'mi300x',
      decodeTp: 4,
    }
    const p = stateToParams(state)
    expect(p.get('sc')).toBe('llmd')
    expect(p.get('mfu')).toBeNull()
    expect(p.get('beff')).toBeNull()
    expect(p.get('beta')).toBeNull()
    expect(stateFromParams(p)).toEqual(state)
  })

  it('llm-d knobs round-trip when changed and reject out-of-range values', () => {
    const state: CalculatorState = { ...DEFAULT_STATE, scenario: 'llmd', llmdMfu: 0.55, llmdBwEff: 0.8, llmdKvBeta: 1.9 }
    const parsed = stateFromParams(stateToParams(state))
    expect(parsed.llmdMfu).toBe(0.55)
    expect(parsed.llmdBwEff).toBe(0.8)
    expect(parsed.llmdKvBeta).toBe(1.9)
    const bad = stateFromParams(new URLSearchParams('sc=llmd&mfu=7&beff=-1&beta=99&rate=-5'))
    expect(bad.llmdMfu).toBe(DEFAULT_STATE.llmdMfu)
    expect(bad.llmdBwEff).toBe(DEFAULT_STATE.llmdBwEff)
    expect(bad.llmdKvBeta).toBe(DEFAULT_STATE.llmdKvBeta)
    expect(bad.requestRate).toBe(DEFAULT_STATE.requestRate)
  })

  it('inference and training URLs stay free of llm-d params', () => {
    expect(stateToParams(DEFAULT_STATE).get('rate')).toBeNull()
    expect(stateToParams({ ...DEFAULT_STATE, scenario: 'training' }).get('pgpu')).toBeNull()
  })

  it('a pre-training-release inference URL still parses (defaults fill the new fields)', () => {
    const p = new URLSearchParams('model=Llama+3.1+8B&w=bf16&kv=fp16&ctx=8192&seqs=8&tp=1&pp=1&gpu=h100-sxm')
    const s = stateFromParams(p)
    expect(s.scenario).toBe('inference')
    expect(s.zeroStage).toBe(DEFAULT_STATE.zeroStage)
    expect(s.dataParallel).toBe(DEFAULT_STATE.dataParallel)
  })

  it('rejects garbage enum values instead of propagating them', () => {
    const p = new URLSearchParams('sc=training&opt=evil&tprec=fp64&zero=9&ckpt=maybe&ft=dora&rank=-3&targ=everything')
    const s = stateFromParams(p)
    expect(s.optimizer).toBe(DEFAULT_STATE.optimizer)
    expect(s.trainPrecision).toBe(DEFAULT_STATE.trainPrecision)
    expect(s.zeroStage).toBe(DEFAULT_STATE.zeroStage)
    expect(s.checkpointing).toBe(DEFAULT_STATE.checkpointing)
    expect(s.ftMethod).toBe('full')
    expect(s.loraRank).toBe(DEFAULT_STATE.loraRank)
    expect(s.loraTargets).toBe(DEFAULT_STATE.loraTargets)
  })
})
