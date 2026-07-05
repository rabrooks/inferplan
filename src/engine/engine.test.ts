import { describe, expect, it } from 'vitest'
import { MODEL_PRESETS } from '../data/models'
import { GPU_DATABASE } from '../data/gpus'
import { GiB, estimateInference, kvBytesPerToken } from './inference'
import { solveMaxTokens } from './fit'
import { kvFormat, weightFormat } from './precision'
import { estimateParams, parseHFConfig } from './hf'
import type { ModelArchitecture } from './types'

const model = (name: string): ModelArchitecture => {
  const m = MODEL_PRESETS.find((m) => m.name === name)
  if (!m) throw new Error(`missing preset ${name}`)
  return m
}
const gpu = (id: string) => GPU_DATABASE.find((g) => g.id === id)!

const quantOf = (w: string, kv: string) => ({ weights: weightFormat(w), kvCache: kvFormat(kv) })
const singleGpu = { tensorParallel: 1, pipelineParallel: 1, replicas: 1 }

describe('weights', () => {
  it('Llama 3.1 8B in BF16 is ~15 GiB of weights', () => {
    const est = estimateInference(
      model('Llama 3.1 8B'),
      quantOf('bf16', 'fp16'),
      { contextLength: 1, concurrentSequences: 1 },
      singleGpu,
    )
    const weights = est.components.find((c) => c.id === 'weights')!
    expect(weights.bytesPerGpu / GiB).toBeCloseTo(14.96, 1)
  })

  it('INT4 weights are ~28% of BF16 (metadata overhead included)', () => {
    const m = model('Llama 3.3 70B')
    const w = { contextLength: 1, concurrentSequences: 1 }
    const bf16 = estimateInference(m, quantOf('bf16', 'fp16'), w, singleGpu)
    const int4 = estimateInference(m, quantOf('int4', 'fp16'), w, singleGpu)
    const ratio =
      int4.components.find((c) => c.id === 'weights')!.bytesPerGpu /
      bf16.components.find((c) => c.id === 'weights')!.bytesPerGpu
    expect(ratio).toBeCloseTo(4.5 / 16, 3)
  })

  it('MoE weights use total params, not active (Mixtral 8x7B > 40 GiB in bf16)', () => {
    const est = estimateInference(
      model('Mixtral 8x7B'),
      quantOf('bf16', 'fp16'),
      { contextLength: 1, concurrentSequences: 1 },
      singleGpu,
    )
    expect(est.components.find((c) => c.id === 'weights')!.bytesPerGpu / GiB).toBeGreaterThan(80)
  })
})

describe('KV cache', () => {
  it('Llama 3.1 8B GQA: 2*32*8*128*2B = 131072 bytes/token at fp16', () => {
    expect(kvBytesPerToken(model('Llama 3.1 8B'), 16)).toBe(131072)
  })

  it('FP8 KV cache halves the fp16 KV size', () => {
    const m = model('Llama 3.3 70B')
    expect(kvBytesPerToken(m, 8)).toBe(kvBytesPerToken(m, 16) / 2)
  })

  it('DeepSeek MLA caches (512+64) elems/layer, far less than equivalent GQA', () => {
    const ds = model('DeepSeek-V3 / R1 671B (MoE, MLA)')
    // 61 layers * 576 elems * 2 bytes = 70,272 bytes/token
    expect(kvBytesPerToken(ds, 16)).toBe(61 * 576 * 2)
  })
})

describe('tensor parallelism', () => {
  it('TP=4 quarters the weights per GPU', () => {
    const m = model('Llama 3.3 70B')
    const w = { contextLength: 8192, concurrentSequences: 4 }
    const tp1 = estimateInference(m, quantOf('bf16', 'fp16'), w, singleGpu)
    const tp4 = estimateInference(m, quantOf('bf16', 'fp16'), w, { ...singleGpu, tensorParallel: 4 })
    expect(tp4.components.find((c) => c.id === 'weights')!.bytesPerGpu).toBeCloseTo(
      tp1.components.find((c) => c.id === 'weights')!.bytesPerGpu / 4,
    )
  })

  it('KV stops sharding beyond the KV-head count and warns', () => {
    const m = model('Llama 3.3 70B') // 8 KV heads
    const w = { contextLength: 8192, concurrentSequences: 4 }
    const tp8 = estimateInference(m, quantOf('bf16', 'fp16'), w, { ...singleGpu, tensorParallel: 8 })
    const tp16 = estimateInference(m, quantOf('bf16', 'fp16'), w, { ...singleGpu, tensorParallel: 16 })
    expect(tp16.components.find((c) => c.id === 'kvCache')!.bytesPerGpu).toBeCloseTo(
      tp8.components.find((c) => c.id === 'kvCache')!.bytesPerGpu,
    )
    expect(tp16.warnings.some((w) => w.includes('KV heads'))).toBe(true)
  })

  it('warns when attention heads are not divisible by TP', () => {
    const est = estimateInference(
      model('Llama 3.1 8B'), // 32 heads
      quantOf('bf16', 'fp16'),
      { contextLength: 1024, concurrentSequences: 1 },
      { ...singleGpu, tensorParallel: 3 },
    )
    expect(est.warnings.some((w) => w.includes('divisible'))).toBe(true)
  })
})

describe('fit and inverse solver', () => {
  it('Llama 3.3 70B bf16 does not fit one H100 but its INT4 quant does', () => {
    const m = model('Llama 3.3 70B')
    const w = { contextLength: 4096, concurrentSequences: 1 }
    const h100 = gpu('h100-sxm')
    const usable = h100.vramGiB * GiB * 0.9
    expect(estimateInference(m, quantOf('bf16', 'fp16'), w, singleGpu).totalBytesPerGpu).toBeGreaterThan(usable)
    expect(estimateInference(m, quantOf('int4', 'fp8'), w, singleGpu).totalBytesPerGpu).toBeLessThan(usable)
  })

  it('solver returns 0 tokens when weights alone overflow the GPU', () => {
    const r = solveMaxTokens(
      model('Llama 3.1 405B'),
      quantOf('bf16', 'fp16'),
      singleGpu,
      gpu('h100-sxm'),
      { contextLength: 8192, concurrentSequences: 1 },
    )
    expect(r.maxTotalTokens).toBe(0)
  })

  it('solver max context is consistent with the forward calculation', () => {
    const m = model('Llama 3.1 8B')
    const q = quantOf('bf16', 'fp16')
    const w = { contextLength: 8192, concurrentSequences: 1 }
    const h100 = gpu('h100-sxm')
    const r = solveMaxTokens(m, q, singleGpu, h100, w)
    const atMax = estimateInference(m, q, { contextLength: r.maxContextAtConcurrency, concurrentSequences: 1 }, singleGpu)
    expect(atMax.totalBytesPerGpu).toBeLessThanOrEqual(h100.vramGiB * GiB * 0.9)
    // and 5% more context should not fit
    const over = estimateInference(
      m,
      q,
      { contextLength: Math.ceil(r.maxContextAtConcurrency * 1.05), concurrentSequences: 1 },
      singleGpu,
    )
    expect(over.totalBytesPerGpu).toBeGreaterThan(h100.vramGiB * GiB * 0.9)
  })
})

describe('HF config parsing', () => {
  it('parses a Llama-3.1-8B-style config and estimates params within 2%', () => {
    const arch = parseHFConfig('meta-llama/Llama-3.1-8B', {
      num_hidden_layers: 32,
      hidden_size: 4096,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      vocab_size: 128256,
      intermediate_size: 14336,
      max_position_embeddings: 131072,
    })
    expect(arch.attentionType).toBe('gqa')
    expect(arch.headDim).toBe(128)
    expect(Math.abs(arch.paramsTotal - 8.03e9) / 8.03e9).toBeLessThan(0.02)
  })

  it('detects MoE and computes active < total (Mixtral-style config)', () => {
    const arch = parseHFConfig('mistralai/Mixtral-8x7B', {
      num_hidden_layers: 32,
      hidden_size: 4096,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      vocab_size: 32000,
      intermediate_size: 14336,
      num_local_experts: 8,
      num_experts_per_tok: 2,
    })
    expect(arch.moe?.numExperts).toBe(8)
    expect(Math.abs(arch.paramsTotal - 46.7e9) / 46.7e9).toBeLessThan(0.02)
    expect(arch.paramsActive).toBeLessThan(arch.paramsTotal / 3)
  })

  it('detects MLA from kv_lora_rank (DeepSeek-style config)', () => {
    const arch = parseHFConfig('deepseek-ai/DeepSeek-V3', {
      num_hidden_layers: 61,
      hidden_size: 7168,
      num_attention_heads: 128,
      vocab_size: 129280,
      intermediate_size: 18432,
      kv_lora_rank: 512,
      qk_rope_head_dim: 64,
      n_routed_experts: 256,
      num_experts_per_tok: 8,
      moe_intermediate_size: 2048,
    })
    expect(arch.attentionType).toBe('mla')
    expect(arch.mla).toEqual({ kvLoraRank: 512, qkRopeHeadDim: 64 })
  })

  it('estimateParams matches published counts for dense presets', () => {
    for (const name of ['Llama 3.1 8B', 'Qwen2.5 72B', 'Mistral 7B v0.3']) {
      const m = model(name)
      const est = estimateParams(m, false)
      expect(Math.abs(est - m.paramsTotal) / m.paramsTotal).toBeLessThan(0.03)
    }
  })
})
