import { estimateInference, kvBytesPerToken } from './inference'
import { DEFAULT_MEMORY_UTILIZATION, solveMaxTokens } from './fit'
import type {
  GPUSpec,
  MemoryEstimate,
  ModelArchitecture,
  QuantizationConfig,
} from './types'

/**
 * llm-d capacity planning: size the prefill and decode pools of a
 * disaggregated deployment from a request rate and latency SLOs.
 *
 * The model is a first-principles roofline that doubles as a latency
 * model — the same physics yields throughput, TTFT, and TPOT:
 *
 * - Prefill is compute-bound: time = FLOPs / (pod TFLOPS × MFU), sized to
 *   a TTFT SLO with an Erlang C (M/M/c) queue-wait term.
 * - Decode is memory-bandwidth-bound: TPOT = bytes streamed per step / BW,
 *   sized to a TPOT SLO and capped by KV-cache capacity.
 *
 * Efficiency assumptions are explicit knobs (LlmdKnobs) rather than baked
 * constants so an InferLens trace can replace each default with a measured
 * value — see docs/interop.md.
 */

/** The request workload the fleet must sustain. */
export interface LlmdWorkload {
  /** Mean request arrival rate λ, requests per second. */
  requestRate: number
  /** Mean prompt length, tokens. */
  inputTokens: number
  /** Mean generated length, tokens. */
  outputTokens: number
  /** Time-to-first-token SLO, ms (queue wait + prefill + KV transfer). */
  ttftSloMs: number
  /** Time-per-output-token SLO, ms. */
  tpotSloMs: number
}

/** One pool's pod shape: which GPU, and how each pod shards the model. */
export interface LlmdPoolInput {
  gpu: GPUSpec
  tensorParallel: number
  pipelineParallel: number
}

/**
 * Documented efficiency heuristics. Each is calibratable from a measured
 * trace (docs/interop.md "Calibration" table).
 */
export interface LlmdKnobs {
  /** Model FLOPs utilization for dense prefill (research band 0.4–0.5). */
  prefillMFU: number
  /** Fraction of spec-sheet HBM bandwidth achieved in decode (band 0.5–0.8). */
  decodeBandwidthEfficiency: number
  /** TTFT multiplier for prefill→decode KV transfer (published 1.8–1.9×). */
  kvTransferBeta: number
  /** Hard per-pool utilization cap ρ for queueing stability (field consensus). */
  maxUtilization: number
}

export const DEFAULT_LLMD_KNOBS: LlmdKnobs = {
  prefillMFU: 0.45,
  decodeBandwidthEfficiency: 0.7,
  kvTransferBeta: 1.8,
  maxUtilization: 0.85,
}

// ---------------------------------------------------------------------------
// Prefill physics (compute-bound)
// ---------------------------------------------------------------------------

/**
 * FLOPs to prefill T prompt tokens: the 2·P_active·T GEMM term (Kaplan
 * 2·W per token; active params for MoE) plus the causal attention term
 * 2·T²·(heads·headDim)·layers. The attention term is always included —
 * negligible at short context, dominant at long (~84% of FLOPs for
 * Llama3-405B at 1M tokens).
 */
export function prefillFlops(model: ModelArchitecture, inputTokens: number): number {
  const attnWidth = model.numAttentionHeads * model.headDim
  const gemm = 2 * model.paramsActive * inputTokens
  const attention = 2 * inputTokens ** 2 * attnWidth * model.numLayers
  return gemm + attention
}

/** Raw compute time for one prefill on one pod of `gpusPerPod` GPUs, seconds. */
export function prefillSeconds(
  model: ModelArchitecture,
  inputTokens: number,
  gpu: GPUSpec,
  gpusPerPod: number,
  mfu: number,
): number {
  return prefillFlops(model, inputTokens) / (gpusPerPod * gpu.bf16TFLOPS * 1e12 * mfu)
}

// ---------------------------------------------------------------------------
// Decode physics (bandwidth-bound)
// ---------------------------------------------------------------------------

/** KV shards across TP — same rule as estimateInference (GQA replication, MLA replicated). */
function kvShardsOf(model: ModelArchitecture, tp: number): number {
  return model.attentionType === 'mla' ? 1 : Math.min(tp, model.numKVHeads)
}

/** Bytes of KV cache one GPU streams per decode step for ONE sequence of `kvLenTokens`. */
export function decodeKvBytesPerSeqPerGpu(
  model: ModelArchitecture,
  quant: QuantizationConfig,
  pool: LlmdPoolInput,
  kvLenTokens: number,
): number {
  // Reuse the memory formula: bytes stored per token are also bytes read per step.
  const perToken = kvBytesPerToken(model, quant.kvCache.bitsPerParam)
  return (perToken * kvLenTokens) / kvShardsOf(model, pool.tensorParallel) / pool.pipelineParallel
}

/**
 * Weight bytes one GPU streams per decode step at batch size n.
 *
 * Dense models stream every weight once per step regardless of batch.
 * MoE models stream only the experts the batch actually routes to: at
 * n=1 that is the active parameters; as n grows the expected expert
 * coverage E·(1−(1−k/E)^n) approaches all experts. This is what makes
 * MoE batched decode cheaper per pod than a dense model of equal total
 * size — and why batch-1 MoE decode is so much faster.
 */
export function decodeWeightBytesPerStepPerGpu(
  model: ModelArchitecture,
  quant: QuantizationConfig,
  pool: LlmdPoolInput,
  n: number,
): number {
  const gpusPerPod = pool.tensorParallel * pool.pipelineParallel
  const totalBytes = (model.paramsTotal * quant.weights.bitsPerParam) / 8
  if (!model.moe || model.moe.numExperts <= model.moe.expertsPerToken) {
    return totalBytes / gpusPerPod
  }
  const { numExperts: E, expertsPerToken: k } = model.moe
  const activeBytes = (model.paramsActive * quant.weights.bitsPerParam) / 8
  // Expected distinct experts touched by n routed tokens, rescaled so
  // coverage(1) = 0 (exactly active params) and coverage(∞) = 1 (all experts).
  const expectedExperts = E * (1 - (1 - k / E) ** n)
  const coverage = Math.min(1, Math.max(0, (expectedExperts - k) / (E - k)))
  return (activeBytes + (totalBytes - activeBytes) * coverage) / gpusPerPod
}

/**
 * Time for one decode step of a batch of n sequences (mean KV length
 * `kvLenTokens`), seconds. This is the TPOT at that concurrency: every
 * GPU streams its weight shard once plus each sequence's KV shard.
 * The linear-in-n shape is fleet-sim's t_iter(n) = W + H·n with W and H
 * derived from GPUSpec physics instead of hand calibration.
 */
export function decodeStepSeconds(
  model: ModelArchitecture,
  quant: QuantizationConfig,
  pool: LlmdPoolInput,
  n: number,
  kvLenTokens: number,
  bandwidthEfficiency: number,
): number {
  const bw = pool.gpu.memoryBandwidthGBs * 1e9 * bandwidthEfficiency
  const weights = decodeWeightBytesPerStepPerGpu(model, quant, pool, n)
  const kv = n * decodeKvBytesPerSeqPerGpu(model, quant, pool, kvLenTokens)
  return (weights + kv) / bw
}

// ---------------------------------------------------------------------------
// Queueing (M/M/c, Erlang C)
// ---------------------------------------------------------------------------

/**
 * Erlang C: probability an arrival must queue, for c servers at offered
 * load a = λ·E[S] (in erlangs). Uses the numerically stable Erlang B
 * recursion B(k) = a·B(k−1)/(k + a·B(k−1)), then C = c·B/(c − a·(1−B)).
 */
export function erlangCWaitProbability(c: number, a: number): number {
  if (a <= 0) return 0
  if (a >= c) return 1 // unstable queue — certain wait
  let b = 1
  for (let k = 1; k <= c; k++) b = (a * b) / (k + a * b)
  return (c * b) / (c - a * (1 - b))
}

/** Mean M/M/c queue wait W_q in seconds. Infinite when the queue is unstable. */
export function erlangCMeanWaitSeconds(c: number, a: number, serviceSeconds: number): number {
  if (a >= c) return Infinity
  return (erlangCWaitProbability(c, a) * serviceSeconds) / (c - a)
}

// ---------------------------------------------------------------------------
// The planner
// ---------------------------------------------------------------------------

export interface LlmdPrefillPlan {
  pods: number
  gpusPerPod: number
  totalGpus: number
  /** Offered-load utilization ρ = λ·E[S] / pods. */
  utilization: number
  flopsPerRequest: number
  /** Raw prefill compute time on one pod, ms. */
  serviceMs: number
  /** Predicted mean queued time (Erlang C), ms — vLLM queued_time. */
  queueMs: number
  /** Predicted prefill phase incl. KV-transfer factor β, ms — vLLM prefill_time. */
  prefillMs: number
  /** queueMs + prefillMs; compare to the TTFT SLO. */
  ttftMs: number
  /** Per-pod memory at one in-flight prefill. */
  memory: MemoryEstimate
}

export interface LlmdDecodePlan {
  pods: number
  gpusPerPod: number
  totalGpus: number
  /** numRunningReqs / maxConcurrency. */
  utilization: number
  /** Predicted steady-state running requests per pod — vLLM num_running_reqs. */
  numRunningReqs: number
  /** Predicted KV usage fraction of capacity (0–1) — vLLM kv_cache_usage. */
  kvCacheUsage: number
  /** Per-pod concurrency ceiling and which constraint set it. */
  maxConcurrency: number
  concurrencyBound: 'tpot-slo' | 'kv-capacity'
  /** Predicted TPOT at the steady-state batch, ms. */
  tpotMs: number
  /** Predicted whole-request decode phase time (outputTokens × TPOT), ms — vLLM decode_time. */
  decodeMs: number
  tokensPerSecondPerPod: number
  tokensPerSecondPerGpu: number
  /** Per-pod memory at the predicted steady-state batch. */
  memory: MemoryEstimate
}

export interface LlmdPlan {
  feasible: boolean
  prefill: LlmdPrefillPlan
  decode: LlmdDecodePlan
  totalGpus: number
  warnings: string[]
}

const MAX_PREFILL_PODS = 100_000

/**
 * Size both pools of a disaggregated llm-d deployment.
 *
 * Prefill pods are M/M/c servers handling one prefill at a time; the pod
 * count is the smallest c with ρ ≤ maxUtilization AND mean queue wait +
 * β×prefill ≤ the TTFT SLO. Decode pods run continuous batching; the pod
 * count is the smallest p whose steady-state batch (Little's law fixed
 * point n = (λ/p)·T_out·t_step(n)) stays under maxUtilization of the
 * per-pod concurrency ceiling min(TPOT-SLO bound, KV-capacity bound).
 *
 * Decode KV lengths use the steady-state mean context T_in + T_out/2:
 * requests in a continuously batched pod are uniformly spread through
 * their generations, and paged KV allocates on demand.
 */
export function planLlmd(
  model: ModelArchitecture,
  quant: QuantizationConfig,
  workload: LlmdWorkload,
  prefillPool: LlmdPoolInput,
  decodePool: LlmdPoolInput,
  knobs: LlmdKnobs = DEFAULT_LLMD_KNOBS,
): LlmdPlan {
  const warnings: string[] = []
  let feasible = true
  const λ = Math.max(0, workload.requestRate)

  // --- Prefill pool ---
  const prefillGpus = prefillPool.tensorParallel * prefillPool.pipelineParallel
  const prefillDeploy = {
    tensorParallel: prefillPool.tensorParallel,
    pipelineParallel: prefillPool.pipelineParallel,
    replicas: 1,
  }
  const prefillMemory = estimateInference(
    model,
    quant,
    { contextLength: workload.inputTokens, concurrentSequences: 1 },
    prefillDeploy,
  )
  const prefillUsable = prefillPool.gpu.vramGiB * 1024 ** 3 * DEFAULT_MEMORY_UTILIZATION
  if (prefillMemory.totalBytesPerGpu > prefillUsable) {
    feasible = false
    warnings.push(
      `Prefill pod does not fit: one prefill of ${workload.inputTokens.toLocaleString()} tokens needs more than ${prefillPool.gpu.name} offers at TP=${prefillPool.tensorParallel}×PP=${prefillPool.pipelineParallel}. Increase the pod's parallelism or quantize.`,
    )
  }

  const flops = prefillFlops(model, workload.inputTokens)
  const service = prefillSeconds(model, workload.inputTokens, prefillPool.gpu, prefillGpus, knobs.prefillMFU)
  const prefillPhaseS = knobs.kvTransferBeta * service
  const sloS = workload.ttftSloMs / 1000

  let prefillPods = Math.max(1, Math.ceil((λ * service) / knobs.maxUtilization))
  let queueS = 0
  if (prefillPhaseS > sloS) {
    feasible = false
    prefillPods = 0
    warnings.push(
      `TTFT SLO unreachable: KV-transfer-adjusted prefill alone is ${Math.round(prefillPhaseS * 1000)} ms > ${workload.ttftSloMs} ms. A prefill pod needs more GPUs (larger TP), a faster GPU, or a shorter prompt.`,
    )
  } else {
    const a = λ * service
    while (prefillPods < MAX_PREFILL_PODS) {
      queueS = erlangCMeanWaitSeconds(prefillPods, a, service)
      if (queueS + prefillPhaseS <= sloS) break
      prefillPods++
    }
  }
  const prefillRho = prefillPods > 0 ? (λ * service) / prefillPods : 0

  const prefill: LlmdPrefillPlan = {
    pods: prefillPods,
    gpusPerPod: prefillGpus,
    totalGpus: prefillPods * prefillGpus,
    utilization: prefillRho,
    flopsPerRequest: flops,
    serviceMs: service * 1000,
    queueMs: queueS * 1000,
    prefillMs: prefillPhaseS * 1000,
    ttftMs: (queueS + prefillPhaseS) * 1000,
    memory: prefillMemory,
  }

  // --- Decode pool ---
  const decodeGpus = decodePool.tensorParallel * decodePool.pipelineParallel
  const decodeDeploy = {
    tensorParallel: decodePool.tensorParallel,
    pipelineParallel: decodePool.pipelineParallel,
    replicas: 1,
  }
  const meanKvLen = workload.inputTokens + workload.outputTokens / 2
  const step = (n: number) =>
    decodeStepSeconds(model, quant, decodePool, n, meanKvLen, knobs.decodeBandwidthEfficiency)

  // KV-capacity concurrency bound, from the existing inverse solver.
  const kvSolve = solveMaxTokens(
    model,
    quant,
    decodeDeploy,
    decodePool.gpu,
    { contextLength: Math.max(1, Math.round(meanKvLen)), concurrentSequences: 1 },
  )
  const nKv = kvSolve.maxConcurrencyAtContext

  // TPOT-SLO concurrency bound: largest integer batch whose step time meets
  // the SLO. Computed independently of the KV bound so we can report which
  // constraint actually binds. step(n) is monotone in n (linear for dense,
  // saturating-expert-coverage + linear KV for MoE), so exponential
  // expansion + bisection is exact.
  const tpotSloS = workload.tpotSloMs / 1000
  const N_SLO_CEILING = 1e7
  let nSlo = 0
  if (step(1) <= tpotSloS) {
    let hi = 2
    while (hi < N_SLO_CEILING && step(hi) <= tpotSloS) hi *= 2
    if (hi >= N_SLO_CEILING) {
      nSlo = hi
    } else {
      let lo = hi / 2
      while (lo < hi - 1) {
        const mid = Math.floor((lo + hi) / 2)
        if (step(mid) <= tpotSloS) lo = mid
        else hi = mid
      }
      nSlo = lo
    }
  }

  const nCap = Math.min(nSlo, nKv)
  const concurrencyBound: LlmdDecodePlan['concurrencyBound'] = nSlo <= nKv ? 'tpot-slo' : 'kv-capacity'
  let decodePods = 0
  let nStar = 0

  if (nKv < 1) {
    feasible = false
    warnings.push(
      `Decode pod has no KV-cache room: weights leave no space for even one sequence of ${Math.round(meanKvLen).toLocaleString()} tokens on ${decodePool.gpu.name} at TP=${decodePool.tensorParallel}×PP=${decodePool.pipelineParallel}.`,
    )
  } else if (nCap < 1) {
    feasible = false
    warnings.push(
      `TPOT SLO unreachable: a single-sequence decode step is already ${(step(1) * 1000).toFixed(1)} ms > ${workload.tpotSloMs} ms. A decode pod needs more GPUs (larger TP), a faster-memory GPU, or lighter quantization.`,
    )
  } else {
    // Size pods so the steady-state batch sits at ≤ maxUtilization of the cap,
    // then solve Little's law n = (λ/p)·T_out·t_step(n) for the actual batch.
    const nTarget = knobs.maxUtilization * nCap
    const tokensNeeded = λ * workload.outputTokens
    decodePods = Math.max(1, Math.ceil((tokensNeeded * step(nTarget)) / nTarget))
    const littles = (n: number) => (λ / decodePods) * workload.outputTokens * step(n)
    let lo = 0
    let hi = nCap
    for (let i = 0; i < 60; i++) {
      const mid = (lo + hi) / 2
      if (mid - littles(mid) < 0) lo = mid
      else hi = mid
    }
    nStar = (lo + hi) / 2
  }

  const tpotS = nStar > 0 ? step(nStar) : step(Math.max(1, nCap))
  const kvUsage = nKv >= 1 && kvSolve.maxTotalTokens > 0 ? (nStar * meanKvLen) / kvSolve.maxTotalTokens : 0
  const decodeMemory = estimateInference(
    model,
    quant,
    {
      contextLength: Math.max(1, Math.round(meanKvLen)),
      concurrentSequences: Math.max(1, Math.round(nStar)),
    },
    decodeDeploy,
  )

  // Large-batch caveat: decode can trend compute-bound; the bandwidth
  // roofline is optimistic once step compute rivals the memory time.
  if (nStar > 0) {
    const computeS =
      (2 * model.paramsActive * nStar) /
      (decodeGpus * decodePool.gpu.bf16TFLOPS * 1e12 * knobs.prefillMFU)
    if (computeS > 0.5 * tpotS) {
      warnings.push(
        `Decode batch of ~${Math.round(nStar)} is approaching the compute roofline on ${decodePool.gpu.name}; the bandwidth-only TPOT estimate is optimistic at this batch size.`,
      )
    }
  }

  const decode: LlmdDecodePlan = {
    pods: decodePods,
    gpusPerPod: decodeGpus,
    totalGpus: decodePods * decodeGpus,
    utilization: nCap >= 1 ? nStar / nCap : 0,
    numRunningReqs: nStar,
    kvCacheUsage: kvUsage,
    maxConcurrency: nCap,
    concurrencyBound,
    tpotMs: tpotS * 1000,
    decodeMs: workload.outputTokens * tpotS * 1000,
    tokensPerSecondPerPod: nStar > 0 ? nStar / tpotS : 0,
    tokensPerSecondPerGpu: nStar > 0 ? nStar / tpotS / decodeGpus : 0,
    memory: decodeMemory,
  }

  for (const w of [...prefillMemory.warnings, ...decodeMemory.warnings]) {
    if (!warnings.includes(w)) warnings.push(w)
  }

  return {
    feasible,
    prefill,
    decode,
    totalGpus: prefill.totalGpus + decode.totalGpus,
    warnings,
  }
}
