import type { DeploymentConfig, GPUSpec, MemoryComponent, MemoryEstimate, ModelArchitecture } from './types'
import { FRAMEWORK_OVERHEAD_BYTES, GiB, fmtParams } from './inference'
import { DEFAULT_MEMORY_UTILIZATION } from './fit'

export type TrainingOptimizer = 'adamw' | 'adamw-8bit' | 'sgd'
export type TrainingPrecision = 'mixed-bf16' | 'fp32'
export type ZeroStage = 0 | 1 | 2 | 3
/**
 * Activation recomputation mode. 'selective' recomputes only the
 * attention-score tensors (what FlashAttention gives for free); 'full'
 * recomputes every layer from its checkpointed input.
 */
export type ActivationCheckpointing = 'none' | 'selective' | 'full'

export type FinetuneMethod = 'full' | 'lora' | 'qlora'
/**
 * Which projections carry LoRA adapters. 'attn-qv' is the LoRA paper's
 * GPT-3 setup (Hu et al. 2021) and the PEFT default for Llama-family
 * models; 'all-linear' is the QLoRA recipe — Dettmers et al. found
 * adapters on every linear layer are needed to match full fine-tuning.
 */
export type LoraTargets = 'attn-qv' | 'attn-all' | 'all-linear'

export interface FinetuneConfig {
  method: FinetuneMethod
  loraRank: number
  loraTargets: LoraTargets
}

export interface TrainingConfig {
  optimizer: TrainingOptimizer
  precision: TrainingPrecision
  zeroStage: ZeroStage
  checkpointing: ActivationCheckpointing
  /** Absent (or method 'full') means full fine-tuning / pre-training. */
  finetune?: FinetuneConfig
}

export interface TrainingWorkload {
  /** Tokens per sequence. */
  sequenceLength: number
  /** Sequences resident on one GPU per step (micro-batch, before grad accumulation). */
  microBatchSize: number
}

export const ZERO_STAGE_NOTES: Record<ZeroStage, string> = {
  0: 'DDP — every rank holds everything',
  1: 'optimizer states sharded across ranks',
  2: 'optimizer + gradients sharded',
  3: 'optimizer + gradients + parameters sharded (≈ FSDP full-shard)',
}

/**
 * Effective bits per parameter of a QLoRA base model: 4-bit NF4 weights,
 * block-64 fp32 quantization scales double-quantized to fp8 with fp32
 * constants per 256 blocks — 4 + 8/64 + 32/(64·256) ≈ 4.127 bits
 * (Dettmers et al., QLoRA §3: DQ cuts scale overhead from 0.5 to 0.127
 * bits/param, ~3 GB on a 65B model). Embeddings and lm_head stay 16-bit
 * in practice; treating them as quantized underestimates by <2%.
 */
export const QLORA_BASE_BITS_PER_PARAM = 4 + 8 / 64 + 32 / (64 * 256)

export const LORA_TARGET_LABELS: Record<LoraTargets, string> = {
  'attn-qv': 'attention Q + V',
  'attn-all': 'all attention (Q, K, V, O)',
  'all-linear': 'all linear layers',
}

/**
 * Trainable adapter parameters: each targeted projection W ∈ R^(d_in×d_out)
 * gains rank-r factors of r·(d_in + d_out) params, summed over layers.
 * GQA-aware (K/V project to numKVHeads·headDim); for MoE models the MLP
 * targets count every expert's projections.
 */
export function loraAdapterParams(model: ModelArchitecture, rank: number, targets: LoraTargets): number {
  const h = model.hiddenSize
  const attnDim = model.numAttentionHeads * model.headDim
  const kvDim = model.numKVHeads * model.headDim
  let perLayer = (h + attnDim) + (h + kvDim) // Q and V
  if (targets !== 'attn-qv') perLayer += (h + kvDim) + (attnDim + h) // K and O
  if (targets === 'all-linear') {
    const ffn = model.moe ? model.moe.expertIntermediateSize : model.intermediateSize
    const experts = model.moe ? model.moe.numExperts : 1
    perLayer += experts * 3 * (h + ffn) // gate, up, down (d_in+d_out is symmetric)
  }
  return rank * perLayer * model.numLayers
}

/**
 * Fixed per-GPU overhead. QLoRA adds a transient dequantization buffer:
 * bitsandbytes dequantizes each NF4 tensor to bf16 for its matmul
 * (QLoRA §3), so the peak holds the largest linear layer in bf16.
 */
export function trainingOverheadBytes(model: ModelArchitecture, config: TrainingConfig): number {
  if (config.finetune?.method !== 'qlora') return FRAMEWORK_OVERHEAD_BYTES
  const h = model.hiddenSize
  const largestLinear = Math.max(
    h * model.numAttentionHeads * model.headDim,
    h * (model.moe ? model.moe.expertIntermediateSize : model.intermediateSize),
  )
  return FRAMEWORK_OVERHEAD_BYTES + 2 * largestLinear
}

/**
 * Bytes per parameter for each persistent training state.
 *
 * Mixed-precision keeps bf16 weights/gradients plus an fp32 master copy of
 * the weights inside the optimizer; AdamW adds fp32 momentum and variance.
 * The mixed AdamW total is the standard 2+2+12 = 16 B/param
 * (EleutherAI transformer-math; George614/gpu-mem-calculator agrees).
 */
export function trainingBytesPerParam(config: TrainingConfig): {
  weights: number
  gradients: number
  optimizer: number
} {
  const mixed = config.precision === 'mixed-bf16'
  // The optimizer's own state, excluding the fp32 master copy:
  // AdamW m+v in fp32 (8), 8-bit m+v via bitsandbytes (2), SGD momentum (4).
  const optimizerState: Record<TrainingOptimizer, number> = {
    adamw: 8,
    'adamw-8bit': 2,
    sgd: 4,
  }
  return {
    weights: mixed ? 2 : 4,
    gradients: mixed ? 2 : 4,
    optimizer: optimizerState[config.optimizer] + (mixed ? 4 : 0),
  }
}

/**
 * Peak activation memory per GPU, from Korthikanti et al. ("Reducing
 * Activation Recomputation in Large Transformer Models"), at TP=1:
 *
 *   none:      s·b·h·L·(34 + 5·a·s/h)   bytes
 *   selective: s·b·h·L·34               bytes  (attention scores recomputed / flash)
 *   full:      s·b·h·L·2                bytes  (only layer inputs retained)
 *
 * Constants assume 2-byte activations and a standard 4h MLP; SwiGLU models
 * differ slightly — this is the published heuristic, documented in
 * docs/FORMULAS.md. fp32 training doubles it.
 */
export function trainingActivationBytes(
  model: ModelArchitecture,
  workload: TrainingWorkload,
  config: TrainingConfig,
): number {
  const { sequenceLength: s, microBatchSize: b } = workload
  const { hiddenSize: h, numLayers: L, numAttentionHeads: a } = model
  const perLayerFactor =
    config.checkpointing === 'full' ? 2 : config.checkpointing === 'selective' ? 34 : 34 + (5 * a * s) / h
  const precisionScale = config.precision === 'fp32' ? 2 : 1
  return s * b * h * L * perLayerFactor * precisionScale
}

/**
 * Per-GPU bytes of the persistent states after ZeRO sharding across `dp`
 * ranks. With LoRA/QLoRA the base weights are frozen — they still occupy
 * memory (bf16, or NF4 for QLoRA) but only the adapter parameters carry
 * gradients and optimizer states. ZeRO/FSDP shards frozen params too, so
 * the stage rules are unchanged.
 */
export function trainingStateBytesPerGpu(
  model: ModelArchitecture,
  config: TrainingConfig,
  dp: number,
): { weights: number; gradients: number; optimizer: number } {
  const per = trainingBytesPerParam(config)
  const P = model.paramsTotal
  const ft = config.finetune
  let weightBytes = P * per.weights
  let trainable = P
  if (ft && ft.method !== 'full') {
    const adapter = loraAdapterParams(model, ft.loraRank, ft.loraTargets)
    const baseBytesPerParam = ft.method === 'qlora' ? QLORA_BASE_BITS_PER_PARAM / 8 : per.weights
    weightBytes = P * baseBytesPerParam + adapter * per.weights
    trainable = adapter
  }
  return {
    weights: weightBytes / (config.zeroStage >= 3 ? dp : 1),
    gradients: (trainable * per.gradients) / (config.zeroStage >= 2 ? dp : 1),
    optimizer: (trainable * per.optimizer) / (config.zeroStage >= 1 ? dp : 1),
  }
}

const OPTIMIZER_LABELS: Record<TrainingOptimizer, string> = {
  adamw: 'AdamW',
  'adamw-8bit': 'AdamW 8-bit',
  sgd: 'SGD (momentum)',
}

/**
 * Estimate GPU memory for training a model with data parallelism.
 *
 * `deploy.replicas` is the data-parallel world size (the total GPU count);
 * ZeRO stages control which states shard across it. Tensor/pipeline
 * parallelism is not modeled for training yet.
 */
export function estimateTraining(
  model: ModelArchitecture,
  config: TrainingConfig,
  workload: TrainingWorkload,
  deploy: DeploymentConfig,
): MemoryEstimate {
  const warnings: string[] = []
  const dp = Math.max(1, deploy.replicas)

  if (deploy.tensorParallel > 1 || deploy.pipelineParallel > 1) {
    warnings.push('Training estimates model data parallelism only; TP/PP settings are ignored.')
  }
  if (model.moe) {
    warnings.push(
      `MoE training is simplified: all ${fmtParams(model.paramsTotal)} params are treated as dense (expert parallelism is not modeled) and the activation estimate uses the shared hidden size.`,
    )
  }
  if (config.zeroStage > 0 && dp === 1) {
    warnings.push(`ZeRO-${config.zeroStage} shards across data-parallel ranks — with a single GPU it changes nothing.`)
  }
  const ft = config.finetune && config.finetune.method !== 'full' ? config.finetune : undefined
  if (ft && model.attentionType === 'mla') {
    warnings.push(
      'Adapter sizing assumes standard Q/K/V/O projections; MLA models factorize attention differently, so the adapter count is approximate.',
    )
  }
  if (ft?.method === 'qlora' && config.precision === 'fp32') {
    warnings.push('QLoRA computes in bf16 — FP32 here only inflates adapter states and activations beyond practice.')
  }

  const per = trainingBytesPerParam(config)
  const state = trainingStateBytesPerGpu(model, config, dp)
  const actPerGpu = trainingActivationBytes(model, workload, config)
  const adapterParams = ft ? loraAdapterParams(model, ft.loraRank, ft.loraTargets) : 0
  const overheadBytes = trainingOverheadBytes(model, config)
  const mixed = config.precision === 'mixed-bf16'
  const shardNote = (stage: ZeroStage) => (config.zeroStage >= stage && dp > 1 ? ` ÷ ${dp} ranks (ZeRO-${config.zeroStage})` : '')

  const ckptLabel =
    config.checkpointing === 'full'
      ? 'full recomputation (2·s·b·h·L)'
      : config.checkpointing === 'selective'
        ? 'selective recomputation (34·s·b·h·L)'
        : 'no recomputation (34 + 5·a·s/h per layer)'

  // Stacking order doubles as the gauge's stratum adjacency: the optimizer
  // magenta sits between the weights blue and the gradients violet because
  // blue↔violet are near-identical under protanopia (ΔE 2.5 in dark mode;
  // 42+ with the magenta between — dataviz validate_palette.js).
  const components: MemoryComponent[] = [
    {
      id: 'weights',
      label: 'Model weights',
      bytesPerGpu: state.weights,
      detail: ft
        ? `${fmtParams(model.paramsTotal)} frozen base × ${
            ft.method === 'qlora' ? `${QLORA_BASE_BITS_PER_PARAM.toFixed(2)} bits (NF4 + double quant)` : `${per.weights} B (${mixed ? 'bf16' : 'fp32'})`
          } + ${fmtParams(adapterParams)} adapter × ${per.weights} B${shardNote(3)}`
        : `${fmtParams(model.paramsTotal)} params × ${per.weights} B (${mixed ? 'bf16' : 'fp32'})${shardNote(3)}`,
    },
    {
      id: 'optimizer',
      label: 'Optimizer states',
      bytesPerGpu: state.optimizer,
      detail: `${OPTIMIZER_LABELS[config.optimizer]}: ${mixed ? 'fp32 master 4 B + ' : ''}${
        config.optimizer === 'adamw' ? 'fp32 m+v 8 B' : config.optimizer === 'adamw-8bit' ? '8-bit m+v 2 B' : 'fp32 momentum 4 B'
      } per param${ft ? ` on ${fmtParams(adapterParams)} adapter params only` : ''}${shardNote(1)}${
        ft?.method === 'qlora' ? ' — bitsandbytes paged, spikes spill to CPU' : ''
      }`,
    },
    {
      id: 'gradients',
      label: 'Gradients',
      bytesPerGpu: state.gradients,
      detail: ft
        ? `${fmtParams(adapterParams)} adapter params × ${per.gradients} B (${mixed ? 'bf16' : 'fp32'}) — frozen base carries none${shardNote(2)}`
        : `${fmtParams(model.paramsTotal)} params × ${per.gradients} B (${mixed ? 'bf16' : 'fp32'})${shardNote(2)}`,
    },
    {
      id: 'activations',
      label: 'Activations (peak, est.)',
      bytesPerGpu: actPerGpu,
      detail: `${workload.sequenceLength.toLocaleString()} tokens × micro-batch ${workload.microBatchSize} × ${model.numLayers} layers — ${ckptLabel}${
        ft ? ' (frozen base still runs forward/backward)' : ''
      }`,
    },
    {
      id: 'overhead',
      label: 'Framework overhead',
      bytesPerGpu: overheadBytes,
      detail: `CUDA context, NCCL, allocator bookkeeping (~0.75 GiB/GPU)${
        ft?.method === 'qlora' ? ' + bf16 dequant buffer for the largest NF4 layer' : ''
      }`,
    },
  ]

  const totalBytesPerGpu = components.reduce((s, c) => s + c.bytesPerGpu, 0)
  return {
    components,
    totalBytesPerGpu,
    totalBytes: totalBytesPerGpu * dp,
    gpusPerReplica: dp,
    warnings,
  }
}

/**
 * Inverse solver for training: with the persistent states fixed, how much
 * activation headroom is left per GPU, and what micro-batch / sequence
 * length does it allow?
 */
export function solveTrainingLimits(
  model: ModelArchitecture,
  config: TrainingConfig,
  workload: TrainingWorkload,
  gpu: GPUSpec,
  dp: number,
  memoryUtilization: number = DEFAULT_MEMORY_UTILIZATION,
): { freeForActivations: number; maxMicroBatch: number; maxSequenceLength: number } {
  const state = trainingStateBytesPerGpu(model, config, Math.max(1, dp))
  const usable = gpu.vramGiB * GiB * memoryUtilization
  const free = usable - state.weights - state.gradients - state.optimizer - trainingOverheadBytes(model, config)
  if (free <= 0) return { freeForActivations: 0, maxMicroBatch: 0, maxSequenceLength: 0 }

  const perSequence = trainingActivationBytes(model, { ...workload, microBatchSize: 1 }, config)
  const maxMicroBatch = Math.floor(free / perSequence)

  // Activation bytes grow monotonically (quadratically without
  // recomputation) in sequence length — binary search the boundary.
  const fits = (s: number) => trainingActivationBytes(model, { ...workload, sequenceLength: s }, config) <= free
  let maxSequenceLength = 0
  if (fits(1)) {
    let lo = 1
    let hi = 1
    while (fits(hi * 2) && hi < 2 ** 24) hi *= 2
    lo = hi
    hi = hi * 2
    while (lo + 1 < hi) {
      const mid = Math.floor((lo + hi) / 2)
      if (fits(mid)) lo = mid
      else hi = mid
    }
    maxSequenceLength = lo
  }
  return { freeForActivations: free, maxMicroBatch, maxSequenceLength }
}

export interface MinTrainFitResult {
  count: number
}

/**
 * Smallest power-of-two data-parallel GPU count that trains the model at
 * this workload. With ZeRO-0 (plain DDP) nothing shards, so per-GPU memory
 * is constant — either one GPU fits or none do.
 */
export function minGpusToTrain(
  model: ModelArchitecture,
  config: TrainingConfig,
  workload: TrainingWorkload,
  gpu: GPUSpec,
  memoryUtilization: number = DEFAULT_MEMORY_UTILIZATION,
  maxCount = 1024,
): MinTrainFitResult | null {
  const usable = gpu.vramGiB * GiB * memoryUtilization
  for (let count = 1; count <= maxCount; count *= 2) {
    const est = estimateTraining(model, config, workload, {
      tensorParallel: 1,
      pipelineParallel: 1,
      replicas: count,
    })
    if (est.totalBytesPerGpu <= usable) return { count }
    if (config.zeroStage === 0) return null
  }
  return null
}
