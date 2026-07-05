/**
 * Core engine types.
 *
 * These types are shared across all calculator scenarios (single-GPU
 * inference, multi-GPU inference, training, fine-tuning, and llm-d
 * capacity planning), so changes here should stay scenario-agnostic.
 */

/** Attention mechanism — determines how the KV cache is sized. */
export type AttentionType = 'mha' | 'gqa' | 'mla'

export interface ModelArchitecture {
  /** Display name, e.g. "Llama 3.1 70B" */
  name: string
  /** Total parameter count (all experts for MoE). */
  paramsTotal: number
  /** Parameters active per token (=== paramsTotal for dense models). */
  paramsActive: number
  numLayers: number
  hiddenSize: number
  numAttentionHeads: number
  /** KV heads; < numAttentionHeads means GQA. */
  numKVHeads: number
  /** Usually hiddenSize / numAttentionHeads, but explicit in some configs. */
  headDim: number
  vocabSize: number
  intermediateSize: number
  attentionType: AttentionType
  /** MoE fields — undefined for dense models. */
  moe?: {
    numExperts: number
    expertsPerToken: number
    /** Intermediate size of each expert's FFN. */
    expertIntermediateSize: number
  }
  /** MLA fields (DeepSeek-style latent attention) — required when attentionType === 'mla'. */
  mla?: {
    kvLoraRank: number
    qkRopeHeadDim: number
  }
  /** Default context window of the model, used to cap workload inputs. */
  maxContextLength?: number
}

/**
 * A numeric-precision format. `bitsPerParam` is the *effective* storage
 * cost including quantization metadata (scales/zero-points), which is why
 * e.g. INT4 group quantization is slightly above 4.0.
 */
export interface PrecisionFormat {
  id: string
  label: string
  bitsPerParam: number
  /** Short note shown in the UI, e.g. "GPTQ/AWQ group-128". */
  note?: string
  /** Whether activations are also quantized (e.g. W8A8) — affects activation memory. */
  activationBits?: number
}

export interface GPUSpec {
  id: string
  name: string
  vendor: 'nvidia' | 'amd' | 'intel'
  /** Usable device memory in GiB. */
  vramGiB: number
  /** Memory bandwidth in GB/s — used for decode throughput estimates (Phase 3+). */
  memoryBandwidthGBs: number
  /** Dense BF16/FP16 TFLOPS — used for prefill/compute estimates (Phase 3+). */
  bf16TFLOPS: number
}

/** How the model is deployed across hardware. Phase 1 uses tensorParallel=1, pipelineParallel=1. */
export interface DeploymentConfig {
  tensorParallel: number
  pipelineParallel: number
  /** Independent replicas of the whole model (Phase 3: llm-d decode/prefill pools). */
  replicas: number
}

/** The serving workload. Phase 3 extends this with request rates. */
export interface InferenceWorkload {
  /** Max tokens in context per sequence (prompt + generation). */
  contextLength: number
  /** Concurrent sequences held in the KV cache. */
  concurrentSequences: number
}

export interface QuantizationConfig {
  /** Weight storage format. */
  weights: PrecisionFormat
  /** KV-cache element format (independent of weights, e.g. FP8 KV). */
  kvCache: PrecisionFormat
}

/** One labelled slice of GPU memory. All values are bytes. */
export interface MemoryComponent {
  id: 'weights' | 'kvCache' | 'activations' | 'gradients' | 'optimizer' | 'overhead'
  label: string
  bytesPerGpu: number
  /** Human explanation of how this number was computed. */
  detail: string
}

export interface MemoryEstimate {
  components: MemoryComponent[]
  /** Sum of components, per GPU. */
  totalBytesPerGpu: number
  /** Total across all GPUs in one replica. */
  totalBytes: number
  gpusPerReplica: number
  /** Non-fatal issues, e.g. "attention heads not divisible by TP". */
  warnings: string[]
}
