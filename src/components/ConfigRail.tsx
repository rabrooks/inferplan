import { useState } from 'react'
import { fetchHFModel } from '../engine/hf'
import { fmtParams } from '../engine/inference'
import { KV_FORMATS, WEIGHT_FORMATS, weightFormat } from '../engine/precision'
import type { ModelArchitecture } from '../engine/types'
import { MODEL_PRESETS } from '../data/models'
import type { CalculatorState } from '../state/url'

interface Props {
  state: CalculatorState
  model: ModelArchitecture
  update: (patch: Partial<CalculatorState>) => void
}

const CTX_PRESETS = [4096, 8192, 32768, 131072]
const SEQ_PRESETS = [1, 8, 32, 64]
const TP_OPTIONS = [1, 2, 4, 8, 16]
const PP_OPTIONS = [1, 2, 4, 8]

export function ConfigRail({ state, model, update }: Props) {
  const [hfInput, setHfInput] = useState('')
  const [hfBusy, setHfBusy] = useState(false)
  const [hfError, setHfError] = useState<string | null>(null)

  const importFromHF = async () => {
    if (!hfInput.trim() || hfBusy) return
    setHfBusy(true)
    setHfError(null)
    try {
      const arch = await fetchHFModel(hfInput)
      update({ modelName: 'custom', customModel: arch })
    } catch (e) {
      setHfError(e instanceof Error ? e.message : String(e))
    } finally {
      setHfBusy(false)
    }
  }

  const wf = weightFormat(state.weightFormat)

  return (
    <div className="rail">
      <section className="section">
        <div className="eyebrow">MODEL</div>
        <div className="field">
          <select
            aria-label="Model preset"
            value={state.modelName}
            onChange={(e) => update({ modelName: e.target.value })}
          >
            {state.customModel && <option value="custom">{state.customModel.name} (imported)</option>}
            {MODEL_PRESETS.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label className="field-label" htmlFor="hf-import">
            Import from Hugging Face
          </label>
          <div className="import-row">
            <input
              id="hf-import"
              type="text"
              placeholder="org/model-name"
              value={hfInput}
              onChange={(e) => setHfInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && importFromHF()}
            />
            <button className="import-btn" onClick={importFromHF} disabled={hfBusy}>
              {hfBusy ? 'Loading…' : 'Import'}
            </button>
          </div>
          {hfError && <div className="import-error">{hfError}</div>}
        </div>
        <dl className="arch-facts">
          <dt>Parameters</dt>
          <dd>
            {fmtParams(model.paramsTotal)}
            {model.moe ? ` total · ${fmtParams(model.paramsActive)} active` : ''}
          </dd>
          <dt>Layers</dt>
          <dd>{model.numLayers}</dd>
          <dt>Hidden size</dt>
          <dd>{model.hiddenSize}</dd>
          <dt>Attention</dt>
          <dd>
            {model.attentionType.toUpperCase()} · {model.numAttentionHeads}h
            {model.attentionType !== 'mla' ? ` / ${model.numKVHeads}kv` : ''}
          </dd>
          {model.moe && (
            <>
              <dt>Experts</dt>
              <dd>
                {model.moe.numExperts} × top-{model.moe.expertsPerToken}
              </dd>
            </>
          )}
        </dl>
      </section>

      <section className="section">
        <div className="eyebrow">PRECISION</div>
        <div className="field">
          <label className="field-label" htmlFor="w-fmt">
            Weights
          </label>
          <select id="w-fmt" value={state.weightFormat} onChange={(e) => update({ weightFormat: e.target.value })}>
            {WEIGHT_FORMATS.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label} — {f.bitsPerParam} bit
              </option>
            ))}
          </select>
          {wf.note && <div className="quant-note">{wf.note}</div>}
        </div>
        <div className="field">
          <label className="field-label" htmlFor="kv-fmt">
            KV cache
          </label>
          <select id="kv-fmt" value={state.kvFormat} onChange={(e) => update({ kvFormat: e.target.value })}>
            {KV_FORMATS.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label} — {f.bitsPerParam} bit
              </option>
            ))}
          </select>
        </div>
      </section>

      <section className="section">
        <div className="eyebrow">WORKLOAD</div>
        <div className="field">
          <label className="field-label" htmlFor="ctx">
            Context length (tokens per sequence)
          </label>
          <input
            id="ctx"
            type="number"
            min={1}
            value={state.contextLength}
            onChange={(e) => update({ contextLength: Math.max(1, Number(e.target.value) || 1) })}
          />
          <div className="chip-row">
            {CTX_PRESETS.map((v) => (
              <button
                key={v}
                className="chip"
                aria-pressed={state.contextLength === v}
                onClick={() => update({ contextLength: v })}
              >
                {v >= 1024 ? `${v / 1024}k` : v}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <label className="field-label" htmlFor="seqs">
            Concurrent sequences
          </label>
          <input
            id="seqs"
            type="number"
            min={1}
            value={state.concurrentSequences}
            onChange={(e) => update({ concurrentSequences: Math.max(1, Number(e.target.value) || 1) })}
          />
          <div className="chip-row">
            {SEQ_PRESETS.map((v) => (
              <button
                key={v}
                className="chip"
                aria-pressed={state.concurrentSequences === v}
                onClick={() => update({ concurrentSequences: v })}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="section">
        <div className="eyebrow">TOPOLOGY</div>
        <div className="field">
          <span className="field-label">Tensor parallelism</span>
          <div className="seg" role="group" aria-label="Tensor parallelism">
            {TP_OPTIONS.map((v) => (
              <button key={v} aria-pressed={state.tensorParallel === v} onClick={() => update({ tensorParallel: v })}>
                {v}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <span className="field-label">Pipeline parallelism</span>
          <div className="seg" role="group" aria-label="Pipeline parallelism">
            {PP_OPTIONS.map((v) => (
              <button
                key={v}
                aria-pressed={state.pipelineParallel === v}
                onClick={() => update({ pipelineParallel: v })}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}
