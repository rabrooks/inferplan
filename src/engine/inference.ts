import type {
  DeploymentConfig,
  InferenceWorkload,
  MemoryComponent,
  MemoryEstimate,
  ModelArchitecture,
  QuantizationConfig,
} from './types'

export const GiB = 1024 ** 3

/**
 * Per-GPU memory the serving framework itself consumes (CUDA context,
 * NCCL buffers, torch allocator bookkeeping). ~0.75 GiB is a good
 * baseline for vLLM/SGLang on NVIDIA hardware.
 */
export const FRAMEWORK_OVERHEAD_BYTES = 0.75 * GiB

/** KV-cache bytes for ONE token across all layers of one replica. */
export function kvBytesPerToken(model: ModelArchitecture, kvBits: number): number {
  if (model.attentionType === 'mla') {
    if (!model.mla) throw new Error(`${model.name}: attentionType is 'mla' but mla config missing`)
    // MLA caches a compressed latent (kvLoraRank) plus the decoupled RoPE
    // key (qkRopeHeadDim) per layer — K and V share the latent.
    const elemsPerLayer = model.mla.kvLoraRank + model.mla.qkRopeHeadDim
    return (model.numLayers * elemsPerLayer * kvBits) / 8
  }
  // Standard MHA/GQA: K and V, one vector of numKVHeads × headDim per layer.
  const elemsPerLayer = 2 * model.numKVHeads * model.headDim
  return (model.numLayers * elemsPerLayer * kvBits) / 8
}

/**
 * Peak transient activation memory during prefill, per replica.
 *
 * Inference frameworks don't retain per-layer activations; the peak is a
 * handful of hidden-state-sized buffers for the widest in-flight batch of
 * tokens. We model ~6 hidden-state copies (QKV projections, attention
 * output, MLP up/gate) capped at a practical max-batched-token count.
 * This is a heuristic — real peak depends on the serving engine's
 * scheduler — so it is labelled an estimate in the UI.
 */
export function activationBytes(
  model: ModelArchitecture,
  workload: InferenceWorkload,
  activationBits: number,
): number {
  const MAX_BATCHED_TOKENS = 8192 // vLLM default max_num_batched_tokens ballpark
  const tokens = Math.min(workload.contextLength * workload.concurrentSequences, MAX_BATCHED_TOKENS)
  const BUFFER_COPIES = 6
  return (tokens * model.hiddenSize * BUFFER_COPIES * activationBits) / 8
}

/**
 * Estimate GPU memory for serving a model.
 *
 * Handles single-GPU (tp=pp=1) and multi-GPU via tensor/pipeline
 * parallelism. Returns a per-GPU breakdown assuming an even shard
 * (the largest pipeline stage will be slightly above this).
 */
export function estimateInference(
  model: ModelArchitecture,
  quant: QuantizationConfig,
  workload: InferenceWorkload,
  deploy: DeploymentConfig,
): MemoryEstimate {
  const warnings: string[] = []
  const { tensorParallel: tp, pipelineParallel: pp } = deploy
  const gpusPerReplica = tp * pp

  if (model.numAttentionHeads % tp !== 0) {
    warnings.push(
      `${model.numAttentionHeads} attention heads are not divisible by TP=${tp}; most engines will refuse this configuration.`,
    )
  }
  if (model.numLayers % pp !== 0) {
    warnings.push(
      `${model.numLayers} layers are not divisible by PP=${pp}; the largest stage will hold more than the per-GPU estimate shown.`,
    )
  }

  // --- Weights ---
  const weightBytesTotal = (model.paramsTotal * quant.weights.bitsPerParam) / 8
  const weightBytesPerGpu = weightBytesTotal / gpusPerReplica

  // --- KV cache ---
  // TP shards KV across heads. When tp exceeds the KV-head count (GQA),
  // engines replicate KV heads instead — no further per-GPU savings.
  const kvShards = model.attentionType === 'mla' ? 1 : Math.min(tp, model.numKVHeads)
  if (model.attentionType !== 'mla' && tp > model.numKVHeads) {
    warnings.push(
      `TP=${tp} exceeds the ${model.numKVHeads} KV heads; KV cache is replicated beyond TP=${model.numKVHeads} and stops shrinking.`,
    )
  }
  if (model.attentionType === 'mla' && tp > 1) {
    warnings.push('MLA latent KV cache is replicated across TP ranks (as in vLLM); only pipeline parallelism shards it.')
  }
  const totalTokens = workload.contextLength * workload.concurrentSequences
  const kvBytesTotal = kvBytesPerToken(model, quant.kvCache.bitsPerParam) * totalTokens
  const kvBytesPerGpu = kvBytesTotal / kvShards / pp

  // --- Activations ---
  const activationBits = quant.weights.activationBits ?? 16
  const actBytesTotal = activationBytes(model, workload, activationBits)
  // TP shards most activation buffers; PP stages each hold their own
  // in-flight microbatch, so PP does not reduce the per-GPU peak.
  const actBytesPerGpu = actBytesTotal / tp

  const components: MemoryComponent[] = [
    {
      id: 'weights',
      label: 'Model weights',
      bytesPerGpu: weightBytesPerGpu,
      detail: `${fmtParams(model.paramsTotal)} params × ${quant.weights.bitsPerParam} bits ÷ ${gpusPerReplica} GPU(s)`,
    },
    {
      id: 'kvCache',
      label: 'KV cache',
      bytesPerGpu: kvBytesPerGpu,
      detail:
        model.attentionType === 'mla'
          ? `MLA latent: ${model.numLayers} layers × ${model.mla!.kvLoraRank + model.mla!.qkRopeHeadDim} elems × ${totalTokens.toLocaleString()} tokens`
          : `2 × ${model.numLayers} layers × ${model.numKVHeads} KV heads × ${model.headDim} dim × ${totalTokens.toLocaleString()} tokens × ${quant.kvCache.bitsPerParam} bits`,
    },
    {
      id: 'activations',
      label: 'Activations (peak, est.)',
      bytesPerGpu: actBytesPerGpu,
      detail: 'Transient prefill buffers; depends on engine scheduler settings',
    },
    {
      id: 'overhead',
      label: 'Framework overhead',
      bytesPerGpu: FRAMEWORK_OVERHEAD_BYTES,
      detail: 'CUDA context, NCCL, allocator bookkeeping (~0.75 GiB/GPU)',
    },
  ]

  const totalBytesPerGpu = components.reduce((s, c) => s + c.bytesPerGpu, 0)
  return {
    components,
    totalBytesPerGpu,
    totalBytes: totalBytesPerGpu * gpusPerReplica,
    gpusPerReplica,
    warnings,
  }
}

export function fmtParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  return `${(n / 1e6).toFixed(0)}M`
}

export function fmtBytes(bytes: number): string {
  const gib = bytes / GiB
  if (gib >= 1000) return `${(gib / 1024).toFixed(2)} TiB`
  if (gib >= 10) return `${gib.toFixed(0)} GiB`
  if (gib >= 1) return `${gib.toFixed(1)} GiB`
  return `${(bytes / 1024 ** 2).toFixed(0)} MiB`
}
