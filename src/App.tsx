import { useEffect, useMemo, useState } from 'react'
import { estimateInference, fmtBytes, GiB } from './engine/inference'
import { DEFAULT_MEMORY_UTILIZATION, minGpusToFit, solveMaxTokens } from './engine/fit'
import { kvFormat, weightFormat } from './engine/precision'
import { GPU_DATABASE } from './data/gpus'
import { DEFAULT_STATE, resolveModel, stateFromParams, stateToParams, type CalculatorState } from './state/url'
import { ConfigRail } from './components/ConfigRail'
import { VramGauge } from './components/VramGauge'
import { BreakdownTable } from './components/BreakdownTable'
import { FitGrid } from './components/FitGrid'

type Theme = 'dark' | 'light'

/**
 * Theme is a viewer preference, not part of a configuration — it lives in
 * localStorage, never in the shareable URL. The pre-paint script in
 * index.html stamps data-theme on <html> before React mounts; this hook
 * reads that, flips it on toggle, and follows OS changes until the viewer
 * makes an explicit choice.
 */
function useTheme() {
  const [theme, setTheme] = useState<Theme>(() =>
    typeof document !== 'undefined' && document.documentElement.dataset.theme === 'light' ? 'light' : 'dark',
  )

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: light)')
    const followOs = () => {
      try {
        if (localStorage.getItem('inferplan-theme')) return
      } catch {
        /* storage blocked — keep following the OS */
      }
      const next: Theme = mq.matches ? 'light' : 'dark'
      document.documentElement.dataset.theme = next
      setTheme(next)
    }
    mq.addEventListener('change', followOs)
    return () => mq.removeEventListener('change', followOs)
  }, [])

  const toggle = () => {
    setTheme((prev) => {
      const next: Theme = prev === 'dark' ? 'light' : 'dark'
      document.documentElement.dataset.theme = next
      try {
        localStorage.setItem('inferplan-theme', next)
      } catch {
        /* storage blocked — theme still applies for this visit */
      }
      return next
    })
  }

  return { theme, toggle }
}

export default function App() {
  const [state, setState] = useState<CalculatorState>(() =>
    typeof window === 'undefined' ? DEFAULT_STATE : stateFromParams(new URLSearchParams(window.location.search)),
  )
  const [copied, setCopied] = useState(false)
  const { theme, toggle } = useTheme()

  const update = (patch: Partial<CalculatorState>) => setState((s) => ({ ...s, ...patch }))

  useEffect(() => {
    const url = `${window.location.pathname}?${stateToParams(state)}`
    window.history.replaceState(null, '', url)
  }, [state])

  const model = resolveModel(state)
  const quant = useMemo(
    () => ({ weights: weightFormat(state.weightFormat), kvCache: kvFormat(state.kvFormat) }),
    [state.weightFormat, state.kvFormat],
  )
  const workload = { contextLength: state.contextLength, concurrentSequences: state.concurrentSequences }
  const deploy = { tensorParallel: state.tensorParallel, pipelineParallel: state.pipelineParallel, replicas: 1 }

  const estimate = estimateInference(model, quant, workload, deploy)
  const gpu = GPU_DATABASE.find((g) => g.id === state.gpuId) ?? GPU_DATABASE[0]
  const usableBytes = gpu.vramGiB * GiB * DEFAULT_MEMORY_UTILIZATION
  const fits = estimate.totalBytesPerGpu <= usableBytes
  const minFit = fits ? null : minGpusToFit(model, quant, workload, gpu)
  const solver = solveMaxTokens(model, quant, deploy, gpu, workload)

  const share = async () => {
    await navigator.clipboard.writeText(window.location.href)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  const gpuCount = estimate.gpusPerReplica

  return (
    <div className="frame">
      <header className="masthead">
        <div>
          <div className="wordmark">
            INFERPLAN<span> / GPU memory</span>
          </div>
          <div className="masthead-sub">LLM deployment calculator — estimates, not benchmarks</div>
        </div>
        <nav className="scenario-tabs" aria-label="Scenario">
          <button className="scenario-tab" aria-current="true">
            Inference
          </button>
          <button className="scenario-tab" disabled title="Planned — see roadmap">
            Training<span className="soon">SOON</span>
          </button>
          <button className="scenario-tab" disabled title="Planned — see roadmap">
            llm-d<span className="soon">SOON</span>
          </button>
        </nav>
        <button className="share-btn" onClick={share}>
          {copied ? 'LINK COPIED' : 'SHARE CONFIG'}
        </button>
        <button
          className="theme-btn"
          onClick={toggle}
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
        >
          {theme === 'dark' ? (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <circle cx="12" cy="12" r="4.5" />
              <path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M18.7 5.3l-2.1 2.1M7.4 16.6l-2.1 2.1" />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11z" />
            </svg>
          )}
        </button>
      </header>

      <div className="layout">
        <ConfigRail state={state} model={model} update={update} />

        <main className="results">
          <div className="verdict">
            <h1 className="verdict-headline">
              {fits ? (
                <>
                  <span className="ok">Fits</span> on {gpuCount}× {gpu.name}
                </>
              ) : minFit ? (
                <>
                  <span className="no">Needs {minFit.count}×</span> {gpu.name}
                  {minFit.count > 1 && (
                    <>
                      {' '}
                      (TP {minFit.shape.tensorParallel}
                      {minFit.shape.pipelineParallel > 1 ? ` · PP ${minFit.shape.pipelineParallel}` : ''})
                    </>
                  )}
                </>
              ) : (
                <>
                  <span className="no">Does not fit</span> {gpu.name} at any count ≤ 128
                </>
              )}
            </h1>
            <div className="verdict-sub">
              {fmtBytes(estimate.totalBytesPerGpu)} of {fmtBytes(usableBytes)} usable per GPU · {model.name} ·{' '}
              {quant.weights.label} weights
            </div>
          </div>

          {estimate.warnings.length > 0 && (
            <div className="warnings">
              {estimate.warnings.map((w) => (
                <div key={w} className="warning">
                  {w}
                </div>
              ))}
            </div>
          )}

          <div className="panels">
            <div className="panel">
              <div className="panel-title">
                PER-GPU VRAM · {gpu.name.toUpperCase()}
                {gpuCount > 1 ? ` · 1 OF ${gpuCount}` : ''}
              </div>
              <VramGauge
                components={estimate.components}
                capacityGiB={gpu.vramGiB}
                memoryUtilization={DEFAULT_MEMORY_UTILIZATION}
              />
            </div>
            <div className="panel">
              <div className="panel-title">BREAKDOWN</div>
              <BreakdownTable estimate={estimate} />
            </div>
          </div>

          <div className="panel">
            <div className="panel-title">
              CAPACITY LIMITS · {gpuCount}× {gpu.name.toUpperCase()} · TP {state.tensorParallel} · PP{' '}
              {state.pipelineParallel}
            </div>
            {solver.maxTotalTokens === 0 ? (
              <p className="foot" style={{ margin: 0 }}>
                Weights and overhead alone exceed usable VRAM at TP {state.tensorParallel} · PP{' '}
                {state.pipelineParallel} — no room for KV cache. Raise the parallelism, pick a larger GPU, or use a
                smaller weight format.
              </p>
            ) : (
              <div className="solver">
                <div className="solver-stat">
                  <div className="stat-value">{solver.maxTotalTokens.toLocaleString()}</div>
                  <div className="stat-label">max KV-cache tokens across all sequences</div>
                </div>
                <div className="solver-stat">
                  <div className="stat-value">{solver.maxContextAtConcurrency.toLocaleString()}</div>
                  <div className="stat-label">
                    max context length at {state.concurrentSequences} concurrent sequences
                  </div>
                </div>
                <div className="solver-stat">
                  <div className="stat-value">{solver.maxConcurrencyAtContext.toLocaleString()}</div>
                  <div className="stat-label">
                    max concurrent sequences at {state.contextLength.toLocaleString()} tokens
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="panel">
            <div className="panel-title">SMALLEST DEPLOYMENT PER GPU MODEL — CLICK TO INSPECT</div>
            <FitGrid
              model={model}
              quant={quant}
              workload={workload}
              selectedGpuId={gpu.id}
              onSelect={(gpuId) => update({ gpuId })}
            />
          </div>

          <p className="foot">
            Estimates assume a vLLM-style engine with {Math.round(DEFAULT_MEMORY_UTILIZATION * 100)}% memory
            utilization and ~0.75 GiB framework overhead per GPU. Weight sizes include quantization metadata.
            Activation peak is a heuristic — treat totals as ±5%. Formulas and sources are documented in the
            repository.
          </p>
        </main>
      </div>
    </div>
  )
}
