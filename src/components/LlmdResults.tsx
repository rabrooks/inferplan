import { useMemo } from 'react'
import { DEFAULT_MEMORY_UTILIZATION } from '../engine/fit'
import { fmtBytes } from '../engine/inference'
import { planLlmd, type LlmdKnobs, type LlmdWorkload } from '../engine/llmd'
import { GPU_DATABASE } from '../data/gpus'
import type { GPUSpec, ModelArchitecture, QuantizationConfig } from '../engine/types'
import type { CalculatorState } from '../state/url'
import { VramGauge } from './VramGauge'
import { FitGrid } from './FitGrid'

interface Props {
  state: CalculatorState
  model: ModelArchitecture
  quant: QuantizationConfig
  update: (patch: Partial<CalculatorState>) => void
}

const gpuById = (id: string): GPUSpec => GPU_DATABASE.find((g) => g.id === id) ?? GPU_DATABASE[0]

function fmtMs(ms: number): string {
  if (!Number.isFinite(ms)) return '∞'
  if (ms >= 10000) return `${(ms / 1000).toFixed(1)} s`
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`
  if (ms >= 100) return `${Math.round(ms)} ms`
  return `${ms.toFixed(1)} ms`
}

interface SloSegment {
  label: string
  ms: number
  color: string
}

/**
 * A bullet bar: phase segments filling toward a dashed SLO tick — the
 * horizontal cousin of the VRAM gauge's capacity line.
 */
function SloBar({ label, segments, sloMs }: { label: string; segments: SloSegment[]; sloMs: number }) {
  const total = segments.reduce((s, seg) => s + seg.ms, 0)
  const scale = Math.max(sloMs * 1.25, total * 1.05)
  const meets = total <= sloMs
  let cum = 0
  return (
    <div className="slo-row">
      <div className="slo-label">{label}</div>
      <div className="slo-track">
        {segments.map((seg) => {
          const left = (cum / scale) * 100
          cum += seg.ms
          return (
            <div
              key={seg.label}
              className="slo-seg"
              title={`${seg.label} · ${fmtMs(seg.ms)}`}
              style={{ left: `${left}%`, width: `max(calc(${(seg.ms / scale) * 100}% - 2px), 0px)`, background: seg.color }}
            />
          )
        })}
        <div className="slo-tick" style={{ left: `${(sloMs / scale) * 100}%` }} />
      </div>
      <div className="slo-value">
        {fmtMs(total)} <span className="slo-of">of {fmtMs(sloMs)}</span>{' '}
        <span className={meets ? 'slo-meets' : 'slo-misses'}>{meets ? '✓ meets SLO' : '✕ misses SLO'}</span>
      </div>
    </div>
  )
}

export function LlmdResults({ state, model, quant, update }: Props) {
  const workload: LlmdWorkload = {
    requestRate: state.requestRate,
    inputTokens: state.inputTokens,
    outputTokens: state.outputTokens,
    ttftSloMs: state.ttftSloMs,
    tpotSloMs: state.tpotSloMs,
  }
  const knobs: LlmdKnobs = {
    prefillMFU: state.llmdMfu,
    decodeBandwidthEfficiency: state.llmdBwEff,
    kvTransferBeta: state.llmdKvBeta,
    maxUtilization: 0.85,
  }
  const prefillGpu = gpuById(state.prefillGpuId)
  const decodeGpu = gpuById(state.decodeGpuId)

  const plan = useMemo(
    () =>
      planLlmd(
        model,
        quant,
        workload,
        { gpu: prefillGpu, tensorParallel: state.prefillTp, pipelineParallel: 1 },
        { gpu: decodeGpu, tensorParallel: state.decodeTp, pipelineParallel: 1 },
        knobs,
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [model, quant, state],
  )

  const sameGpu = prefillGpu.id === decodeGpu.id
  const { prefill, decode } = plan

  const fleetFor = (g: GPUSpec) => {
    const p = planLlmd(
      model,
      quant,
      workload,
      { gpu: g, tensorParallel: state.prefillTp, pipelineParallel: 1 },
      { gpu: g, tensorParallel: state.decodeTp, pipelineParallel: 1 },
      knobs,
    )
    if (!p.feasible) return null
    return { count: p.totalGpus, shapeLabel: `${p.prefill.pods} pre + ${p.decode.pods} dec pods` }
  }

  return (
    <main className="results">
      <div className="verdict">
        <h1 className="verdict-headline">
          {plan.feasible ? (
            sameGpu ? (
              <>
                <span className="ok">{plan.totalGpus}× {prefillGpu.name}</span> — {prefill.pods} prefill +{' '}
                {decode.pods} decode pods
              </>
            ) : (
              <>
                <span className="ok">{plan.totalGpus} GPUs</span> — {prefill.pods}× {prefillGpu.name} prefill +{' '}
                {decode.pods}× {decodeGpu.name} decode pods
              </>
            )
          ) : (
            <>
              <span className="no">No feasible fleet</span> at this pod shape
            </>
          )}
        </h1>
        <div className="verdict-sub">
          {model.name} · {quant.weights.label} weights · {state.requestRate} req/s ·{' '}
          {state.inputTokens.toLocaleString()} in / {state.outputTokens.toLocaleString()} out
        </div>
      </div>

      {plan.warnings.length > 0 && (
        <div className="warnings">
          {plan.warnings.map((w) => (
            <div key={w} className="warning">
              {w}
            </div>
          ))}
        </div>
      )}

      <div className="panel">
        <div className="panel-title">PREDICTED LATENCY vs SLO · ERLANG C QUEUE + ROOFLINE SERVICE TIMES</div>
        <div className="slo-bars">
          <SloBar
            label="TTFT"
            sloMs={state.ttftSloMs}
            segments={[
              { label: 'queued', ms: prefill.queueMs, color: 'var(--c-ovh)' },
              { label: 'prefill + KV transfer', ms: prefill.prefillMs, color: 'var(--c-weights)' },
            ]}
          />
          <SloBar
            label="TPOT"
            sloMs={state.tpotSloMs}
            segments={[{ label: 'decode step', ms: decode.tpotMs, color: 'var(--c-kv)' }]}
          />
        </div>
        <p className="slo-note">
          TTFT = <span className="swatch" style={{ background: 'var(--c-ovh)' }} />
          queue {fmtMs(prefill.queueMs)} + <span className="swatch" style={{ background: 'var(--c-weights)' }} />
          prefill {fmtMs(prefill.prefillMs)} (raw {fmtMs(prefill.serviceMs)} × β {state.llmdKvBeta} KV transfer) ·
          TPOT at {decode.numRunningReqs.toFixed(1)} running requests per decode pod · full decode phase ≈{' '}
          {fmtMs(decode.decodeMs)} per request
        </p>
      </div>

      <div className="pools">
        <div className="panel">
          <div className="panel-title">PREFILL POOL · COMPUTE-BOUND · {prefillGpu.name.toUpperCase()}</div>
          <div className="pool-head">
            <div className="pool-count">
              {prefill.pods}
              <span className="pool-count-unit"> pods × {prefill.gpusPerPod} GPU{prefill.gpusPerPod > 1 ? 's' : ''}</span>
            </div>
            <div className="pool-total">{prefill.totalGpus} GPUs</div>
          </div>
          <dl className="arch-facts">
            <dt>Utilization ρ</dt>
            <dd>{prefill.utilization.toFixed(2)} of 0.85 cap</dd>
            <dt>Prefill compute</dt>
            <dd>{fmtMs(prefill.serviceMs)} / request</dd>
            <dt>Queue wait (M/M/c)</dt>
            <dd>{fmtMs(prefill.queueMs)} mean</dd>
            <dt>FLOPs / request</dt>
            <dd>{(prefill.flopsPerRequest / 1e12).toFixed(1)} TFLOP</dd>
          </dl>
          <div className="pool-gauge-title">PER-GPU VRAM · ONE POD · 1 PREFILL IN FLIGHT</div>
          <VramGauge
            components={prefill.memory.components}
            capacityGiB={prefillGpu.vramGiB}
            memoryUtilization={DEFAULT_MEMORY_UTILIZATION}
          />
          <div className="pool-gauge-foot">
            {fmtBytes(prefill.memory.totalBytesPerGpu)} of{' '}
            {fmtBytes(prefillGpu.vramGiB * 1024 ** 3 * DEFAULT_MEMORY_UTILIZATION)} usable per GPU
          </div>
        </div>

        <div className="panel">
          <div className="panel-title">DECODE POOL · BANDWIDTH-BOUND · {decodeGpu.name.toUpperCase()}</div>
          <div className="pool-head">
            <div className="pool-count">
              {decode.pods}
              <span className="pool-count-unit"> pods × {decode.gpusPerPod} GPU{decode.gpusPerPod > 1 ? 's' : ''}</span>
            </div>
            <div className="pool-total">{decode.totalGpus} GPUs</div>
          </div>
          <dl className="arch-facts">
            <dt>Utilization ρ</dt>
            <dd>{decode.utilization.toFixed(2)} of 0.85 cap</dd>
            <dt>Running reqs / pod</dt>
            <dd>
              {decode.numRunningReqs.toFixed(1)} of {decode.maxConcurrency} max (
              {decode.concurrencyBound === 'kv-capacity' ? 'KV-bound' : 'TPOT-bound'})
            </dd>
            <dt>KV cache usage</dt>
            <dd>{Math.round(decode.kvCacheUsage * 100)}%</dd>
            <dt>Throughput</dt>
            <dd>{Math.round(decode.tokensPerSecondPerGpu).toLocaleString()} tok/s/GPU</dd>
          </dl>
          <div className="pool-gauge-title">
            PER-GPU VRAM · ONE POD · {Math.max(1, Math.round(decode.numRunningReqs))} RUNNING SEQS
          </div>
          <VramGauge
            components={decode.memory.components}
            capacityGiB={decodeGpu.vramGiB}
            memoryUtilization={DEFAULT_MEMORY_UTILIZATION}
          />
          <div className="pool-gauge-foot">
            {fmtBytes(decode.memory.totalBytesPerGpu)} of{' '}
            {fmtBytes(decodeGpu.vramGiB * 1024 ** 3 * DEFAULT_MEMORY_UTILIZATION)} usable per GPU
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">PREDICTED OBSERVABLES — CHECK AGAINST A LIVE DEPLOYMENT</div>
        <div className="table-scroll">
        <table className="breakdown">
          <thead>
            <tr>
              <th scope="col">QUANTITY</th>
              <th scope="col">PREDICTED</th>
              <th scope="col">VLLM / INFERLENS COUNTER</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Running requests per decode pod</td>
              <td className="num">{decode.numRunningReqs.toFixed(1)}</td>
              <td className="num">SchedulerStats.num_running_reqs</td>
            </tr>
            <tr>
              <td>KV cache usage (decode pod)</td>
              <td className="num">{decode.kvCacheUsage.toFixed(2)}</td>
              <td className="num">SchedulerStats.kv_cache_usage</td>
            </tr>
            <tr>
              <td>Queued / prefill / decode time per request</td>
              <td className="num">
                {fmtMs(prefill.queueMs)} / {fmtMs(prefill.prefillMs)} / {fmtMs(decode.decodeMs)}
              </td>
              <td className="num">FinishedRequestStats</td>
            </tr>
            <tr>
              <td>Decode throughput</td>
              <td className="num">
                {Math.round(decode.tokensPerSecondPerPod).toLocaleString()} tok/s/pod ·{' '}
                {Math.round(decode.tokensPerSecondPerPod * Math.max(1, decode.pods)).toLocaleString()} pool
              </td>
              <td className="num">per-request token counts + timings</td>
            </tr>
          </tbody>
        </table>
        </div>
        <p className="foot" style={{ marginTop: 10 }}>
          InferPlan predicts, <a href="https://github.com/rabrooks/inferlens">InferLens</a> observes: record each pool
          with InferLens and these numbers overlay the trace directly — the contract is{' '}
          <a href="https://github.com/rabrooks/inferplan/blob/main/docs/interop.md">docs/interop.md</a>. Where they
          disagree, calibrate the MFU / bandwidth-efficiency / β knobs from the trace.
        </p>
      </div>

      <div className="panel">
        <div className="panel-title">SLO-MEETING FLEET PER GPU MODEL (BOTH POOLS) — CLICK TO INSPECT</div>
        <FitGrid
          fitFor={fleetFor}
          noFitLabel="SLO not met"
          selectedGpuId={sameGpu ? decodeGpu.id : ''}
          onSelect={(gpuId) => update({ prefillGpuId: gpuId, decodeGpuId: gpuId })}
        />
      </div>

      <p className="foot">
        Fleet sizing is a first-principles roofline: prefill FLOPs ÷ (TFLOPS × MFU {state.llmdMfu}), decode bytes ÷
        (bandwidth × {state.llmdBwEff}), Erlang C mean queue wait, hard ρ ≤ 0.85 per pool, KV transfer as a{' '}
        {state.llmdKvBeta}× TTFT factor. Mean token counts stand in for full arrival/length distributions — treat pod
        counts as a starting point for load testing, not a guarantee. Formulas and sources are documented in the
        repository.
      </p>
    </main>
  )
}
