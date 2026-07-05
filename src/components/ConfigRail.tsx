import { useState } from 'react'
import { fetchHFModel } from '../engine/hf'
import { fmtParams } from '../engine/inference'
import { KV_FORMATS, WEIGHT_FORMATS, weightFormat } from '../engine/precision'
import { ZERO_STAGE_NOTES, loraAdapterParams, type FinetuneMethod, type ZeroStage } from '../engine/training'
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
const SEQ_LEN_PRESETS = [1024, 4096, 8192, 32768]
const MICRO_BATCH_PRESETS = [1, 2, 4, 8]
const DP_PRESETS = [1, 8, 16, 64, 256]
const ZERO_STAGES: ZeroStage[] = [0, 1, 2, 3]

const CKPT_NOTES = {
  none: 'every activation retained — attention cost grows with seq²',
  selective: 'attention scores recomputed (what FlashAttention gives for free)',
  full: 'only layer inputs retained; ~30% extra compute per step',
} as const

const LORA_RANK_PRESETS = [8, 16, 32, 64]

const FT_NOTES: Record<FinetuneMethod, string> = {
  full: 'every parameter carries gradients and optimizer states',
  lora: 'frozen bf16 base; only low-rank adapters train',
  qlora: 'frozen NF4 4-bit base + adapters (Dettmers et al.)',
}

const TARGET_OPTIONS = [
  { id: 'attn-qv', label: 'Attention Q + V — LoRA paper default' },
  { id: 'attn-all', label: 'All attention — Q, K, V, O' },
  { id: 'all-linear', label: 'All linear layers — QLoRA recipe' },
] as const

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
  const training = state.scenario === 'training'

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

      {training ? (
        <section className="section">
          <div className="eyebrow">TRAINING</div>
          <div className="field">
            <span className="field-label">Method</span>
            <div className="seg" role="group" aria-label="Fine-tuning method">
              {(['full', 'lora', 'qlora'] as const).map((v) => (
                <button
                  key={v}
                  aria-pressed={state.ftMethod === v}
                  onClick={() =>
                    update(v === 'qlora' ? { ftMethod: v, trainPrecision: 'mixed-bf16' } : { ftMethod: v })
                  }
                >
                  {v === 'full' ? 'Full FT' : v === 'lora' ? 'LoRA' : 'QLoRA'}
                </button>
              ))}
            </div>
            <div className="quant-note">{FT_NOTES[state.ftMethod]}</div>
          </div>
          {state.ftMethod !== 'full' && (
            <>
              <div className="field">
                <label className="field-label" htmlFor="lora-rank">
                  Adapter rank (r)
                </label>
                <input
                  id="lora-rank"
                  type="number"
                  min={1}
                  value={state.loraRank}
                  onChange={(e) => update({ loraRank: Math.max(1, Number(e.target.value) || 1) })}
                />
                <div className="chip-row">
                  {LORA_RANK_PRESETS.map((v) => (
                    <button
                      key={v}
                      className="chip"
                      aria-pressed={state.loraRank === v}
                      onClick={() => update({ loraRank: v })}
                    >
                      {v}
                    </button>
                  ))}
                </div>
              </div>
              <div className="field">
                <label className="field-label" htmlFor="lora-targ">
                  Adapter targets
                </label>
                <select
                  id="lora-targ"
                  value={state.loraTargets}
                  onChange={(e) => update({ loraTargets: e.target.value as CalculatorState['loraTargets'] })}
                >
                  {TARGET_OPTIONS.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.label}
                    </option>
                  ))}
                </select>
                <div className="quant-note">
                  ≈ {fmtParams(loraAdapterParams(model, state.loraRank, state.loraTargets))} adapter params ·{' '}
                  {((loraAdapterParams(model, state.loraRank, state.loraTargets) / model.paramsTotal) * 100).toFixed(2)}
                  % of base
                </div>
              </div>
            </>
          )}
          <div className="field">
            <span className="field-label">Precision</span>
            <div className="seg" role="group" aria-label="Training precision">
              <button
                aria-pressed={state.trainPrecision === 'mixed-bf16'}
                onClick={() => update({ trainPrecision: 'mixed-bf16' })}
              >
                Mixed BF16
              </button>
              <button
                aria-pressed={state.trainPrecision === 'fp32'}
                disabled={state.ftMethod === 'qlora'}
                title={state.ftMethod === 'qlora' ? 'QLoRA computes in bf16' : undefined}
                onClick={() => update({ trainPrecision: 'fp32' })}
              >
                FP32
              </button>
            </div>
          </div>
          <div className="field">
            <label className="field-label" htmlFor="optim">
              Optimizer
            </label>
            <select
              id="optim"
              value={state.optimizer}
              onChange={(e) => update({ optimizer: e.target.value as CalculatorState['optimizer'] })}
            >
              <option value="adamw">AdamW — fp32 m+v</option>
              <option value="adamw-8bit">AdamW 8-bit — bitsandbytes m+v</option>
              <option value="sgd">SGD — fp32 momentum</option>
            </select>
          </div>
          <div className="field">
            <span className="field-label">ZeRO / FSDP sharding</span>
            <div className="seg" role="group" aria-label="ZeRO stage">
              {ZERO_STAGES.map((v) => (
                <button key={v} aria-pressed={state.zeroStage === v} onClick={() => update({ zeroStage: v })}>
                  {v === 0 ? 'DDP' : `Z${v}`}
                </button>
              ))}
            </div>
            <div className="quant-note">{ZERO_STAGE_NOTES[state.zeroStage]}</div>
          </div>
          <div className="field">
            <span className="field-label">Activation checkpointing</span>
            <div className="seg" role="group" aria-label="Activation checkpointing">
              {(['none', 'selective', 'full'] as const).map((v) => (
                <button key={v} aria-pressed={state.checkpointing === v} onClick={() => update({ checkpointing: v })}>
                  {v[0].toUpperCase() + v.slice(1)}
                </button>
              ))}
            </div>
            <div className="quant-note">{CKPT_NOTES[state.checkpointing]}</div>
          </div>
        </section>
      ) : (
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
      )}

      <section className="section">
        <div className="eyebrow">WORKLOAD</div>
        <div className="field">
          <label className="field-label" htmlFor="ctx">
            {training ? 'Sequence length (tokens)' : 'Context length (tokens per sequence)'}
          </label>
          <input
            id="ctx"
            type="number"
            min={1}
            value={state.contextLength}
            onChange={(e) => update({ contextLength: Math.max(1, Number(e.target.value) || 1) })}
          />
          <div className="chip-row">
            {(training ? SEQ_LEN_PRESETS : CTX_PRESETS).map((v) => (
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
        {training ? (
          <div className="field">
            <label className="field-label" htmlFor="mb">
              Micro-batch size (sequences per GPU)
            </label>
            <input
              id="mb"
              type="number"
              min={1}
              value={state.microBatch}
              onChange={(e) => update({ microBatch: Math.max(1, Number(e.target.value) || 1) })}
            />
            <div className="chip-row">
              {MICRO_BATCH_PRESETS.map((v) => (
                <button
                  key={v}
                  className="chip"
                  aria-pressed={state.microBatch === v}
                  onClick={() => update({ microBatch: v })}
                >
                  {v}
                </button>
              ))}
            </div>
          </div>
        ) : (
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
        )}
      </section>

      <section className="section">
        <div className="eyebrow">TOPOLOGY</div>
        {training ? (
          <div className="field">
            <label className="field-label" htmlFor="dp">
              Data-parallel GPUs (ZeRO shards across these)
            </label>
            <input
              id="dp"
              type="number"
              min={1}
              value={state.dataParallel}
              onChange={(e) => update({ dataParallel: Math.max(1, Number(e.target.value) || 1) })}
            />
            <div className="chip-row">
              {DP_PRESETS.map((v) => (
                <button
                  key={v}
                  className="chip"
                  aria-pressed={state.dataParallel === v}
                  onClick={() => update({ dataParallel: v })}
                >
                  {v}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            <div className="field">
              <span className="field-label">Tensor parallelism</span>
              <div className="seg" role="group" aria-label="Tensor parallelism">
                {TP_OPTIONS.map((v) => (
                  <button
                    key={v}
                    aria-pressed={state.tensorParallel === v}
                    onClick={() => update({ tensorParallel: v })}
                  >
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
          </>
        )}
      </section>
    </div>
  )
}
