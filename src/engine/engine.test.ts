import { describe, expect, it } from 'vitest'
import { MODEL_PRESETS } from '../data/models'
import { GPU_DATABASE } from '../data/gpus'
import { GiB, estimateInference, kvBytesPerToken } from './inference'
import { solveMaxTokens } from './fit'
import { kvFormat, weightFormat } from './precision'
import { estimateParams, parseHFConfig } from './hf'
import {
  QLORA_BASE_BITS_PER_PARAM,
  estimateTraining,
  loraAdapterParams,
  minGpusToTrain,
  solveTrainingLimits,
  trainingActivationBytes,
  trainingBytesPerParam,
  type TrainingConfig,
} from './training'
import {
  DEFAULT_LLMD_KNOBS,
  decodeKvBytesPerSeqPerGpu,
  decodeStepSeconds,
  decodeWeightBytesPerStepPerGpu,
  erlangCMeanWaitSeconds,
  erlangCWaitProbability,
  planLlmd,
  prefillFlops,
  prefillSeconds,
  type LlmdWorkload,
} from './llmd'
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

describe('training states', () => {
  const mixedAdamW: TrainingConfig = {
    optimizer: 'adamw',
    precision: 'mixed-bf16',
    zeroStage: 0,
    checkpointing: 'selective',
  }
  const ddp1 = { tensorParallel: 1, pipelineParallel: 1, replicas: 1 }
  const bytesOf = (est: ReturnType<typeof estimateTraining>, id: string) =>
    est.components.find((c) => c.id === id)!.bytesPerGpu

  it('mixed-precision AdamW is 16 B/param: bf16 weights 2 + bf16 grads 2 + fp32 master 4 + m/v 8 (transformer-math)', () => {
    const per = trainingBytesPerParam(mixedAdamW)
    expect(per.weights + per.gradients + per.optimizer).toBe(16)
    const m = model('Llama 3.1 8B')
    const est = estimateTraining(m, mixedAdamW, { sequenceLength: 1, microBatchSize: 1 }, ddp1)
    expect(bytesOf(est, 'weights')).toBe(2 * m.paramsTotal)
    expect(bytesOf(est, 'gradients')).toBe(2 * m.paramsTotal)
    expect(bytesOf(est, 'optimizer')).toBe(12 * m.paramsTotal)
  })

  it('8-bit AdamW optimizer states are 6 B/param, SGD+momentum 8 B/param (bitsandbytes / transformer-math)', () => {
    expect(trainingBytesPerParam({ ...mixedAdamW, optimizer: 'adamw-8bit' }).optimizer).toBe(6)
    expect(trainingBytesPerParam({ ...mixedAdamW, optimizer: 'sgd' }).optimizer).toBe(8)
  })

  it('fp32 AdamW is also 16 B/param, with no separate master copy: 4 + 4 + 8', () => {
    const per = trainingBytesPerParam({ ...mixedAdamW, precision: 'fp32' })
    expect(per).toEqual({ weights: 4, gradients: 4, optimizer: 8 })
  })

  it('ZeRO stages shard optimizer, then gradients, then parameters across DP ranks', () => {
    const m = model('Llama 3.1 8B')
    const w = { sequenceLength: 1, microBatchSize: 1 }
    const dp8 = { tensorParallel: 1, pipelineParallel: 1, replicas: 8 }
    const z1 = estimateTraining(m, { ...mixedAdamW, zeroStage: 1 }, w, dp8)
    expect(bytesOf(z1, 'optimizer')).toBe((12 * m.paramsTotal) / 8)
    expect(bytesOf(z1, 'gradients')).toBe(2 * m.paramsTotal)
    expect(bytesOf(z1, 'weights')).toBe(2 * m.paramsTotal)
    const z2 = estimateTraining(m, { ...mixedAdamW, zeroStage: 2 }, w, dp8)
    expect(bytesOf(z2, 'gradients')).toBe((2 * m.paramsTotal) / 8)
    expect(bytesOf(z2, 'weights')).toBe(2 * m.paramsTotal)
    const z3 = estimateTraining(m, { ...mixedAdamW, zeroStage: 3 }, w, dp8)
    expect(bytesOf(z3, 'weights')).toBe((2 * m.paramsTotal) / 8)
    expect(bytesOf(z3, 'gradients')).toBe((2 * m.paramsTotal) / 8)
    expect(bytesOf(z3, 'optimizer')).toBe((12 * m.paramsTotal) / 8)
  })

  it('warns when ZeRO is enabled on a single GPU, and for MoE models', () => {
    const w = { sequenceLength: 1, microBatchSize: 1 }
    const z3 = estimateTraining(model('Llama 3.1 8B'), { ...mixedAdamW, zeroStage: 3 }, w, ddp1)
    expect(z3.warnings.some((x) => x.includes('single GPU'))).toBe(true)
    const moe = estimateTraining(model('Mixtral 8x7B'), mixedAdamW, w, ddp1)
    expect(moe.warnings.some((x) => x.includes('MoE'))).toBe(true)
  })
})

describe('training activations', () => {
  const base: TrainingConfig = { optimizer: 'adamw', precision: 'mixed-bf16', zeroStage: 0, checkpointing: 'none' }

  it('no recomputation matches Korthikanti/transformer-math sbhL(34 + 5as/h) at TP=1', () => {
    const m = model('Llama 3.1 8B') // h=4096, L=32, a=32
    const w = { sequenceLength: 4096, microBatchSize: 1 }
    // factor = 34 + 5·32·4096/4096 = 194; sbhL = 4096·1·4096·32
    expect(trainingActivationBytes(m, w, base)).toBe(4096 * 4096 * 32 * 194)
  })

  it('selective recomputation keeps 34·sbhL; full checkpointing keeps 2·sbhL', () => {
    const m = model('Llama 3.1 8B')
    const w = { sequenceLength: 4096, microBatchSize: 1 }
    const sbhL = 4096 * 1 * 4096 * 32
    expect(trainingActivationBytes(m, w, { ...base, checkpointing: 'selective' })).toBe(34 * sbhL)
    expect(trainingActivationBytes(m, w, { ...base, checkpointing: 'full' })).toBe(2 * sbhL)
  })

  it('scales linearly with micro-batch and doubles under fp32', () => {
    const m = model('Llama 3.1 8B')
    const w = { sequenceLength: 2048, microBatchSize: 1 }
    expect(trainingActivationBytes(m, { ...w, microBatchSize: 4 }, base)).toBe(4 * trainingActivationBytes(m, w, base))
    expect(trainingActivationBytes(m, w, { ...base, precision: 'fp32' })).toBe(2 * trainingActivationBytes(m, w, base))
  })
})

describe('training fit and limits', () => {
  const cfg: TrainingConfig = { optimizer: 'adamw', precision: 'mixed-bf16', zeroStage: 3, checkpointing: 'full' }

  it('Llama 3.1 8B mixed AdamW ZeRO-3 needs 4× A100-40 (16 B/param states sharded)', () => {
    const fit = minGpusToTrain(model('Llama 3.1 8B'), cfg, { sequenceLength: 4096, microBatchSize: 1 }, gpu('a100-40'))
    expect(fit?.count).toBe(4)
  })

  it('plain DDP cannot fit by adding GPUs — nothing shards', () => {
    const fit = minGpusToTrain(
      model('Llama 3.3 70B'),
      { ...cfg, zeroStage: 0 },
      { sequenceLength: 4096, microBatchSize: 1 },
      gpu('h100-sxm'),
    )
    expect(fit).toBeNull()
  })

  it('solver micro-batch limit is consistent with the forward estimate', () => {
    const m = model('Llama 3.1 8B')
    const w = { sequenceLength: 4096, microBatchSize: 1 }
    const h100 = gpu('h100-sxm')
    const dp8 = { tensorParallel: 1, pipelineParallel: 1, replicas: 8 }
    const usable = h100.vramGiB * GiB * 0.9
    const { maxMicroBatch } = solveTrainingLimits(m, cfg, w, h100, 8)
    expect(maxMicroBatch).toBeGreaterThan(0)
    const atMax = estimateTraining(m, cfg, { ...w, microBatchSize: maxMicroBatch }, dp8)
    expect(atMax.totalBytesPerGpu).toBeLessThanOrEqual(usable)
    const over = estimateTraining(m, cfg, { ...w, microBatchSize: maxMicroBatch + 1 }, dp8)
    expect(over.totalBytesPerGpu).toBeGreaterThan(usable)
  })

  it('solver sequence-length limit is consistent with the forward estimate (quadratic regime)', () => {
    const m = model('Llama 3.1 8B')
    const noCkpt: TrainingConfig = { ...cfg, checkpointing: 'none' }
    const w = { sequenceLength: 4096, microBatchSize: 1 }
    const h100 = gpu('h100-sxm')
    const dp8 = { tensorParallel: 1, pipelineParallel: 1, replicas: 8 }
    const usable = h100.vramGiB * GiB * 0.9
    const { maxSequenceLength } = solveTrainingLimits(m, noCkpt, w, h100, 8)
    expect(maxSequenceLength).toBeGreaterThan(0)
    const atMax = estimateTraining(m, noCkpt, { sequenceLength: maxSequenceLength, microBatchSize: 1 }, dp8)
    expect(atMax.totalBytesPerGpu).toBeLessThanOrEqual(usable)
    const over = estimateTraining(m, noCkpt, { sequenceLength: maxSequenceLength + 1, microBatchSize: 1 }, dp8)
    expect(over.totalBytesPerGpu).toBeGreaterThan(usable)
  })
})

describe('fine-tuning (LoRA / QLoRA)', () => {
  const mixedAdamW: TrainingConfig = {
    optimizer: 'adamw',
    precision: 'mixed-bf16',
    zeroStage: 0,
    checkpointing: 'full',
  }
  const ddp1 = { tensorParallel: 1, pipelineParallel: 1, replicas: 1 }
  const bytesOf = (est: ReturnType<typeof estimateTraining>, id: string) =>
    est.components.find((c) => c.id === id)!.bytesPerGpu

  // Architectures from the papers the tests are pinned to — not presets.
  const gpt3: ModelArchitecture = {
    name: 'GPT-3 175B',
    paramsTotal: 175e9,
    paramsActive: 175e9,
    numLayers: 96,
    hiddenSize: 12288,
    numAttentionHeads: 96,
    numKVHeads: 96,
    headDim: 128,
    vocabSize: 50257,
    intermediateSize: 49152,
    attentionType: 'mha',
  }
  const llama65b: ModelArchitecture = {
    name: 'LLaMA 65B',
    paramsTotal: 65.2e9,
    paramsActive: 65.2e9,
    numLayers: 80,
    hiddenSize: 8192,
    numAttentionHeads: 64,
    numKVHeads: 64,
    headDim: 128,
    vocabSize: 32000,
    intermediateSize: 22016,
    attentionType: 'mha',
  }

  it('LoRA paper pin: GPT-3 175B, r=4 on Q+V ≈ 18.9M adapters — the ~10,000× trainable-param reduction', () => {
    // Hu et al. 2021: checkpoint drops 350 GB → ~35 MB (fp16), i.e. ~17.5M params.
    const adapters = loraAdapterParams(gpt3, 4, 'attn-qv')
    expect(adapters).toBe(4 * (2 * (12288 + 12288)) * 96) // 18,874,368
    expect(gpt3.paramsTotal / adapters).toBeGreaterThan(9000)
  })

  it('finetune: "full" (or absent) is exactly the Phase 4 estimator', () => {
    const m = model('Llama 3.1 8B')
    const w = { sequenceLength: 4096, microBatchSize: 1 }
    const plain = estimateTraining(m, mixedAdamW, w, ddp1)
    const full = estimateTraining(
      m,
      { ...mixedAdamW, finetune: { method: 'full', loraRank: 16, loraTargets: 'attn-qv' } },
      w,
      ddp1,
    )
    expect(full).toEqual(plain)
  })

  it('LoRA: frozen bf16 base, gradients + optimizer on adapter params only', () => {
    const m = model('Llama 3.1 8B') // h=4096, attn 32×128, kv 8×128, L=32
    const adapters = loraAdapterParams(m, 16, 'attn-qv')
    expect(adapters).toBe(16 * (4096 + 4096 + (4096 + 1024)) * 32) // 6,815,744
    const est = estimateTraining(
      m,
      { ...mixedAdamW, finetune: { method: 'lora', loraRank: 16, loraTargets: 'attn-qv' } },
      { sequenceLength: 4096, microBatchSize: 1 },
      ddp1,
    )
    expect(bytesOf(est, 'weights')).toBe(2 * m.paramsTotal + 2 * adapters)
    expect(bytesOf(est, 'gradients')).toBe(2 * adapters)
    expect(bytesOf(est, 'optimizer')).toBe(12 * adapters)
  })

  it('QLoRA paper pin: LLaMA 65B lands at ~48 GB on one GPU vs >780 GB for 16-bit full FT', () => {
    const w = { sequenceLength: 512, microBatchSize: 1 }
    // Dettmers et al. 2023: full 16-bit finetuning of 65B "requires more than 780 GB".
    const fullFT = estimateTraining(llama65b, mixedAdamW, w, ddp1)
    expect(fullFT.totalBytesPerGpu).toBeGreaterThan(780e9)
    // QLoRA recipe: NF4+DQ base, r=64 adapters on all linear layers, paged 32-bit AdamW.
    const qlora = estimateTraining(
      llama65b,
      { ...mixedAdamW, finetune: { method: 'qlora', loraRank: 64, loraTargets: 'all-linear' } },
      w,
      ddp1,
    )
    const adapters = loraAdapterParams(llama65b, 64, 'all-linear')
    expect(adapters).toBe(64 * (4 * (8192 + 8192) + 3 * (8192 + 22016)) * 80) // ≈0.8B
    // Paper claims "<48GB" average with paging absorbing spikes; we estimate within ±5%.
    expect(qlora.totalBytesPerGpu / 1e9).toBeGreaterThan(43)
    expect(qlora.totalBytesPerGpu / 1e9).toBeLessThan(50.5)
    // The bitsandbytes 8-bit optimizer brings it comfortably under 48 GB.
    const qlora8bit = estimateTraining(
      llama65b,
      { ...mixedAdamW, optimizer: 'adamw-8bit', finetune: { method: 'qlora', loraRank: 64, loraTargets: 'all-linear' } },
      w,
      ddp1,
    )
    expect(qlora8bit.totalBytesPerGpu).toBeLessThan(48e9)
  })

  it('QLoRA base weights are 4.127 effective bits (NF4 + double quantization) plus a dequant buffer', () => {
    expect(QLORA_BASE_BITS_PER_PARAM).toBeCloseTo(4.127, 3)
    const m = model('Llama 3.1 8B')
    const w = { sequenceLength: 4096, microBatchSize: 1 }
    const ft = { loraRank: 16, loraTargets: 'attn-qv' as const }
    const lora = estimateTraining(m, { ...mixedAdamW, finetune: { method: 'lora', ...ft } }, w, ddp1)
    const qlora = estimateTraining(m, { ...mixedAdamW, finetune: { method: 'qlora', ...ft } }, w, ddp1)
    const adapters = loraAdapterParams(m, 16, 'attn-qv')
    expect(bytesOf(qlora, 'weights')).toBeCloseTo((m.paramsTotal * QLORA_BASE_BITS_PER_PARAM) / 8 + 2 * adapters, -1)
    // Dequant transient: largest NF4 linear (h × intermediate) held in bf16.
    expect(bytesOf(qlora, 'overhead') - bytesOf(lora, 'overhead')).toBe(2 * 4096 * 14336)
  })

  it('activations are unchanged by LoRA/QLoRA — the frozen base still runs forward/backward', () => {
    const m = model('Llama 3.1 8B')
    const w = { sequenceLength: 4096, microBatchSize: 1 }
    const full = estimateTraining(m, mixedAdamW, w, ddp1)
    const qlora = estimateTraining(
      m,
      { ...mixedAdamW, finetune: { method: 'qlora', loraRank: 64, loraTargets: 'all-linear' } },
      w,
      ddp1,
    )
    expect(bytesOf(qlora, 'activations')).toBe(bytesOf(full, 'activations'))
  })

  it('ZeRO shards adapter states; the frozen base only shards at stage 3', () => {
    const m = model('Llama 3.1 8B')
    const w = { sequenceLength: 1, microBatchSize: 1 }
    const dp8 = { tensorParallel: 1, pipelineParallel: 1, replicas: 8 }
    const ft = { method: 'lora' as const, loraRank: 16, loraTargets: 'attn-qv' as const }
    const adapters = loraAdapterParams(m, 16, 'attn-qv')
    const z2 = estimateTraining(m, { ...mixedAdamW, zeroStage: 2, finetune: ft }, w, dp8)
    expect(bytesOf(z2, 'optimizer')).toBe((12 * adapters) / 8)
    expect(bytesOf(z2, 'gradients')).toBe((2 * adapters) / 8)
    expect(bytesOf(z2, 'weights')).toBe(2 * m.paramsTotal + 2 * adapters)
    const z3 = estimateTraining(m, { ...mixedAdamW, zeroStage: 3, finetune: ft }, w, dp8)
    expect(bytesOf(z3, 'weights')).toBe((2 * m.paramsTotal + 2 * adapters) / 8)
  })

  it('MoE all-linear targeting counts every expert projection', () => {
    const m = model('Mixtral 8x7B') // 8 experts, expert FFN 14336
    const attn = loraAdapterParams(m, 16, 'attn-all')
    const all = loraAdapterParams(m, 16, 'all-linear')
    expect(all - attn).toBe(16 * 8 * 3 * (4096 + 14336) * 32)
  })

  it('QLoRA fits Llama 3.3 70B on a single H100 where full FT needs a big ZeRO-3 cluster', () => {
    const w = { sequenceLength: 4096, microBatchSize: 1 }
    const qloraFit = minGpusToTrain(
      model('Llama 3.3 70B'),
      { ...mixedAdamW, optimizer: 'adamw-8bit', finetune: { method: 'qlora', loraRank: 16, loraTargets: 'attn-qv' } },
      w,
      gpu('h100-sxm'),
    )
    expect(qloraFit?.count).toBe(1)
    const fullFit = minGpusToTrain(model('Llama 3.3 70B'), { ...mixedAdamW, zeroStage: 3 }, w, gpu('h100-sxm'))
    expect(fullFit?.count).toBeGreaterThanOrEqual(16)
  })

  it('solver headroom reflects adapter-only states', () => {
    const m = model('Llama 3.1 8B')
    const w = { sequenceLength: 4096, microBatchSize: 1 }
    const h100 = gpu('h100-sxm')
    const full = solveTrainingLimits(m, mixedAdamW, w, h100, 1)
    const lora = solveTrainingLimits(
      m,
      { ...mixedAdamW, finetune: { method: 'lora', loraRank: 16, loraTargets: 'attn-qv' } },
      w,
      h100,
      1,
    )
    // Full FT of 8B is ~128 GB of state — no activation room on one 80 GB GPU.
    expect(full.freeForActivations).toBe(0)
    expect(lora.freeForActivations).toBeGreaterThan(50 * GiB)
    expect(lora.maxMicroBatch).toBeGreaterThan(full.maxMicroBatch)
  })
})

describe('llm-d capacity planning', () => {
  const h100 = gpu('h100-sxm')
  const pool = (gpuId: string, tp: number, pp = 1) => ({
    gpu: gpu(gpuId),
    tensorParallel: tp,
    pipelineParallel: pp,
  })

  describe('prefill roofline (compute-bound)', () => {
    it('MoE prefill FLOPs use active params, not total (Mixtral)', () => {
      const m = model('Mixtral 8x7B') // 12.9B active of 46.7B total
      const flops = prefillFlops(m, 1000)
      // 2·P_active·T + 2·T²·(heads·headDim)·L
      expect(flops).toBe(2 * 12.9e9 * 1000 + 2 * 1000 ** 2 * (32 * 128) * 32)
      expect(flops).toBeLessThan(2 * m.paramsTotal * 1000)
    })

    it('Meta measured pin: Llama3-405B 128K prefill on one 8×H100 host ≈ 60 s; our bf16-spec formula at default MFU lands within 20%', () => {
      // arXiv:2411.01783 (Context Parallelism, MLSys'25): ~60 s measured (FP8).
      const t = prefillSeconds(model('Llama 3.1 405B'), 131072, h100, 8, DEFAULT_LLMD_KNOBS.prefillMFU)
      expect(t).toBeGreaterThan(60 * 0.75)
      expect(t).toBeLessThan(60 * 1.05)
    })

    it('Meta measured pin: 405B at 1M tokens on 128 H100s at their measured 63% util ≈ 77 s, attention term is ~84% of FLOPs', () => {
      // Same paper: 1M-token prefill ≈ 77 s at 63% FLOPS utilization.
      const m = model('Llama 3.1 405B')
      const t = prefillSeconds(m, 1048576, h100, 128, 0.63)
      expect(t / 77).toBeGreaterThan(0.85)
      expect(t / 77).toBeLessThan(1.1)
      // The quadratic attention term dominates at 1M context — this is why
      // it is always included rather than gated on a crossover heuristic.
      const attnShare = 1 - (2 * m.paramsActive * 1048576) / prefillFlops(m, 1048576)
      expect(attnShare).toBeCloseTo(0.84, 2)
    })
  })

  describe('decode roofline (bandwidth-bound)', () => {
    it('BentoML handbook pin: 70B FP16 (~140 GB) on H100 3.35 TB/s → ~24 tok/s single-sequence ceiling', () => {
      // bentoml.com/llm/getting-started/choosing-the-right-gpu: "a theoretical
      // ceiling of about 24 tokens/s per sequence" (before KV, eff = 1).
      const m70: ModelArchitecture = {
        name: '70B dense',
        paramsTotal: 70e9,
        paramsActive: 70e9,
        numLayers: 80,
        hiddenSize: 8192,
        numAttentionHeads: 64,
        numKVHeads: 8,
        headDim: 128,
        vocabSize: 128256,
        intermediateSize: 28672,
        attentionType: 'gqa',
      }
      const step = decodeStepSeconds(m70, quantOf('fp16', 'fp16'), pool('h100-sxm', 1), 1, 0, 1.0)
      expect(1 / step).toBeCloseTo(23.9, 1)
    })

    it('fleet-sim cross-check (order of magnitude only): derived W and H for Llama-70B/H100/8K bracket their calibrated 4.0 ms and 0.32 ms/slot', () => {
      // arXiv:2603.16054 calibrates t_iter(n) = W + H·n by hand. Our
      // physically derived values differ (their constants bake in unstated
      // TP/quant/occupancy assumptions) but must land in the same regime.
      const m = model('Llama 3.3 70B')
      const hMs = (decodeKvBytesPerSeqPerGpu(m, quantOf('bf16', 'fp16'), pool('h100-sxm', 1), 8192) / 3.35e12) * 1000
      expect(hMs).toBeCloseTo(0.8, 2)
      expect(hMs / 0.32).toBeLessThan(4)
      const wMs = (decodeWeightBytesPerStepPerGpu(m, quantOf('bf16', 'fp16'), pool('h100-sxm', 8), 1) / 3.35e12) * 1000
      expect(wMs / 4.0).toBeGreaterThan(0.5)
      expect(wMs / 4.0).toBeLessThan(2)
    })

    it('dense weight traffic is batch-independent; KV traffic scales linearly with the batch', () => {
      const m = model('Llama 3.3 70B')
      const q = quantOf('bf16', 'fp16')
      const p = pool('h100-sxm', 2)
      expect(decodeWeightBytesPerStepPerGpu(m, q, p, 1)).toBe(decodeWeightBytesPerStepPerGpu(m, q, p, 64))
      const s1 = decodeStepSeconds(m, q, p, 1, 4096, 0.7)
      const s65 = decodeStepSeconds(m, q, p, 65, 4096, 0.7)
      expect((s65 - s1) / 64).toBeCloseTo(decodeKvBytesPerSeqPerGpu(m, q, p, 4096) / (3.35e12 * 0.7), 12)
    })

    it('MoE expert coverage: batch 1 streams active params only, large batches stream toward all experts (DeepSeek-V3)', () => {
      const ds = model('DeepSeek-V3 / R1 671B (MoE, MLA)')
      const q = quantOf('fp8', 'fp8')
      const p = pool('h200', 8)
      expect(decodeWeightBytesPerStepPerGpu(ds, q, p, 1)).toBeCloseTo(37e9 / 8, 0)
      const large = decodeWeightBytesPerStepPerGpu(ds, q, p, 10000)
      expect(large / (671e9 / 8)).toBeGreaterThan(0.99)
      // monotone in n
      expect(decodeWeightBytesPerStepPerGpu(ds, q, p, 64)).toBeGreaterThan(decodeWeightBytesPerStepPerGpu(ds, q, p, 8))
    })

    it('MLA KV is replicated across TP ranks: per-GPU KV traffic does not shrink with TP', () => {
      const ds = model('DeepSeek-V3 / R1 671B (MoE, MLA)')
      const q = quantOf('fp8', 'fp8')
      expect(decodeKvBytesPerSeqPerGpu(ds, q, pool('h200', 8), 1000)).toBe(
        decodeKvBytesPerSeqPerGpu(ds, q, pool('h200', 1), 1000),
      )
    })
  })

  describe('Erlang C queueing', () => {
    it('reduces to M/M/1: P(wait) = ρ and W_q = ρ/(1−ρ)·s at c=1', () => {
      expect(erlangCWaitProbability(1, 0.5)).toBeCloseTo(0.5, 10)
      expect(erlangCMeanWaitSeconds(1, 0.5, 2)).toBeCloseTo(2, 10)
    })

    it('matches the published M/M/2 value: a=1 erlang → P(wait) = 1/3', () => {
      expect(erlangCWaitProbability(2, 1)).toBeCloseTo(1 / 3, 10)
    })

    it('adding servers reduces the wait; an unstable queue waits forever', () => {
      expect(erlangCMeanWaitSeconds(3, 1, 1)).toBeLessThan(erlangCMeanWaitSeconds(2, 1, 1))
      expect(erlangCMeanWaitSeconds(1, 1.2, 1)).toBe(Infinity)
    })
  })

  describe('planLlmd end-to-end', () => {
    // The worked example: Llama 3.3 70B bf16, 100 req/s of 1024-in/256-out,
    // fleet-sim's SLO defaults (TTFT 500 ms, TPOT 100 ms), H100 TP=2 pods.
    const workload: LlmdWorkload = {
      requestRate: 100,
      inputTokens: 1024,
      outputTokens: 256,
      ttftSloMs: 500,
      tpotSloMs: 100,
    }
    const m70 = model('Llama 3.3 70B')
    const q = quantOf('bf16', 'fp16')

    it('sizes the 100 req/s 70B fleet: both SLOs met, both pools under the ρ ≤ 0.85 cap, KV-bound decode', () => {
      const plan = planLlmd(m70, q, workload, pool('h100-sxm', 2), pool('h100-sxm', 2))
      expect(plan.feasible).toBe(true)
      expect(plan.prefill.pods).toBe(20)
      expect(plan.decode.pods).toBe(33)
      expect(plan.prefill.ttftMs).toBeLessThanOrEqual(500)
      expect(plan.decode.tpotMs).toBeLessThanOrEqual(100)
      expect(plan.prefill.utilization).toBeLessThanOrEqual(0.85)
      expect(plan.decode.utilization).toBeLessThanOrEqual(0.85 + 1e-9)
      // 80 GiB of H100 fills with KV long before the 100 ms TPOT budget does.
      expect(plan.decode.concurrencyBound).toBe('kv-capacity')
      expect(plan.totalGpus).toBe(20 * 2 + 33 * 2)
    })

    it('emits the InferLens vocabulary: predicted running requests, KV usage fraction, phase timings', () => {
      const plan = planLlmd(m70, q, workload, pool('h100-sxm', 2), pool('h100-sxm', 2))
      expect(plan.decode.numRunningReqs).toBeCloseTo(24.9, 1)
      expect(plan.decode.kvCacheUsage).toBeCloseTo(0.8, 1)
      expect(plan.decode.kvCacheUsage).toBeLessThan(1)
      expect(plan.prefill.queueMs).toBeGreaterThan(0)
      expect(plan.prefill.prefillMs).toBeCloseTo(1.8 * plan.prefill.serviceMs, 6)
      expect(plan.decode.decodeMs).toBeCloseTo(256 * plan.decode.tpotMs, 6)
    })

    it('steady state is self-consistent: pool decode throughput equals the arriving token rate, per-GPU rate is in the published 70B-on-H100 band', () => {
      const plan = planLlmd(m70, q, workload, pool('h100-sxm', 2), pool('h100-sxm', 2))
      const arriving = workload.requestRate * workload.outputTokens
      expect(plan.decode.tokensPerSecondPerPod * plan.decode.pods).toBeCloseTo(arriving, 4)
      // ~390 tok/s/GPU — consistent with real-world bf16 70B H100 serving rates.
      expect(plan.decode.tokensPerSecondPerGpu).toBeGreaterThan(300)
      expect(plan.decode.tokensPerSecondPerGpu).toBeLessThan(500)
    })

    it('an unreachable TPOT SLO is reported infeasible, not silently sized', () => {
      const plan = planLlmd(m70, q, { ...workload, tpotSloMs: 5 }, pool('h100-sxm', 2), pool('h100-sxm', 2))
      expect(plan.feasible).toBe(false)
      expect(plan.decode.pods).toBe(0)
      expect(plan.warnings.some((w) => w.includes('TPOT SLO unreachable'))).toBe(true)
    })

    it('an unreachable TTFT SLO (prefill alone exceeds it) is reported infeasible', () => {
      const plan = planLlmd(
        model('Llama 3.1 405B'),
        q,
        { ...workload, inputTokens: 131072 },
        pool('h100-sxm', 8),
        pool('h100-sxm', 8),
      )
      expect(plan.feasible).toBe(false)
      expect(plan.warnings.some((w) => w.includes('TTFT SLO unreachable'))).toBe(true)
    })

    it('decode pods whose weights leave no KV room are reported infeasible', () => {
      const plan = planLlmd(model('Llama 3.1 405B'), q, workload, pool('h100-sxm', 8), pool('h100-sxm', 1))
      expect(plan.feasible).toBe(false)
      expect(plan.warnings.some((w) => w.includes('no KV-cache room'))).toBe(true)
    })

    it('zero request rate degenerates to one idle pod per pool', () => {
      const plan = planLlmd(m70, q, { ...workload, requestRate: 0 }, pool('h100-sxm', 2), pool('h100-sxm', 2))
      expect(plan.prefill.pods).toBe(1)
      expect(plan.decode.pods).toBe(1)
      expect(plan.decode.numRunningReqs).toBeCloseTo(0, 6)
      expect(plan.prefill.queueMs).toBe(0)
    })

    it('DeepSeek-V3 demo: MoE active params + MLA KV make an fp8 H200 fleet dramatically smaller than dense-671B physics would suggest', () => {
      const ds = model('DeepSeek-V3 / R1 671B (MoE, MLA)')
      const dsWorkload: LlmdWorkload = {
        requestRate: 20,
        inputTokens: 2048,
        outputTokens: 512,
        ttftSloMs: 500,
        tpotSloMs: 50,
      }
      const plan = planLlmd(ds, quantOf('fp8', 'fp8'), dsWorkload, pool('h200', 8), pool('h200', 8))
      expect(plan.feasible).toBe(true)
      expect(plan.prefill.pods).toBe(2)
      // fp8 MLA KV is so small that one 8×H200 pod holds the whole decode load.
      expect(plan.decode.pods).toBe(1)
      expect(plan.decode.tpotMs).toBeLessThanOrEqual(50)
      // MLA latent KV admits hundreds of concurrent sequences per pod.
      expect(plan.decode.maxConcurrency).toBeGreaterThan(200)
      expect(plan.decode.numRunningReqs).toBeGreaterThan(100)
    })
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
