import { FRAMEWORK_OVERHEAD_BYTES, GiB, activationBytes, estimateInference, kvBytesPerToken } from './inference'
import type {
  DeploymentConfig,
  GPUSpec,
  InferenceWorkload,
  MemoryEstimate,
  ModelArchitecture,
  QuantizationConfig,
} from './types'

/**
 * Fraction of VRAM the serving engine is allowed to use
 * (vLLM's gpu_memory_utilization; leaves headroom for fragmentation).
 */
export const DEFAULT_MEMORY_UTILIZATION = 0.9

export interface GpuFit {
  gpu: GPUSpec
  /** Whether one replica fits on `gpusPerReplica` of this GPU. */
  fits: boolean
  /** Fraction of usable VRAM consumed per GPU (can exceed 1). */
  utilization: number
  /** Smallest count of this GPU that could hold the replica at the same TP×PP shape, if any (power-of-two search hint). */
  freeBytesPerGpu: number
}

/** Check a computed estimate against each GPU in the list. */
export function fitAcrossGpus(
  estimate: MemoryEstimate,
  gpus: GPUSpec[],
  memoryUtilization: number = DEFAULT_MEMORY_UTILIZATION,
): GpuFit[] {
  return gpus.map((gpu) => {
    const usable = gpu.vramGiB * GiB * memoryUtilization
    return {
      gpu,
      fits: estimate.totalBytesPerGpu <= usable,
      utilization: estimate.totalBytesPerGpu / usable,
      freeBytesPerGpu: usable - estimate.totalBytesPerGpu,
    }
  })
}

/**
 * Inverse solver: given a GPU and deployment shape, how many total KV-cache
 * tokens fit? From that, derive max context length at a given concurrency,
 * or max concurrent sequences at a given context length.
 */
export function solveMaxTokens(
  model: ModelArchitecture,
  quant: QuantizationConfig,
  deploy: DeploymentConfig,
  gpu: GPUSpec,
  workload: InferenceWorkload,
  memoryUtilization: number = DEFAULT_MEMORY_UTILIZATION,
): { maxTotalTokens: number; maxContextAtConcurrency: number; maxConcurrencyAtContext: number } {
  const { tensorParallel: tp, pipelineParallel: pp } = deploy
  const gpusPerReplica = tp * pp
  const usablePerGpu = gpu.vramGiB * GiB * memoryUtilization

  const weightsPerGpu = (model.paramsTotal * quant.weights.bitsPerParam) / 8 / gpusPerReplica
  const activationBits = quant.weights.activationBits ?? 16
  const actPerGpu = activationBytes(model, workload, activationBits) / tp
  const freeForKvPerGpu = usablePerGpu - weightsPerGpu - actPerGpu - FRAMEWORK_OVERHEAD_BYTES

  if (freeForKvPerGpu <= 0) {
    return { maxTotalTokens: 0, maxContextAtConcurrency: 0, maxConcurrencyAtContext: 0 }
  }

  const kvShards = model.attentionType === 'mla' ? 1 : Math.min(tp, model.numKVHeads)
  const perTokenPerGpu = kvBytesPerToken(model, quant.kvCache.bitsPerParam) / kvShards / pp
  const maxTotalTokens = Math.floor(freeForKvPerGpu / perTokenPerGpu)

  return {
    maxTotalTokens,
    maxContextAtConcurrency: Math.floor(maxTotalTokens / workload.concurrentSequences),
    maxConcurrencyAtContext: Math.floor(maxTotalTokens / workload.contextLength),
  }
}

export interface MinFitResult {
  count: number
  shape: DeploymentConfig
}

/**
 * Smallest power-of-two count of a given GPU that can serve the model at
 * this workload. Prefers tensor parallelism; when TP would violate
 * head-divisibility, the remainder becomes pipeline stages. Returns null
 * if nothing up to `maxCount` fits.
 */
export function minGpusToFit(
  model: ModelArchitecture,
  quant: QuantizationConfig,
  workload: InferenceWorkload,
  gpu: GPUSpec,
  memoryUtilization: number = DEFAULT_MEMORY_UTILIZATION,
  maxCount = 128,
): MinFitResult | null {
  const usable = gpu.vramGiB * GiB * memoryUtilization
  for (let count = 1; count <= maxCount; count *= 2) {
    let tp = count
    while (tp > 1 && model.numAttentionHeads % tp !== 0) tp /= 2
    const pp = count / tp
    if (pp > model.numLayers) continue
    const shape: DeploymentConfig = { tensorParallel: tp, pipelineParallel: pp, replicas: 1 }
    const est = estimateInference(model, quant, workload, shape)
    if (est.totalBytesPerGpu <= usable) return { count, shape }
  }
  return null
}
