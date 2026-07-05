import type { ModelArchitecture } from './types'

/**
 * Import a model architecture from a Hugging Face repo.
 *
 * Reads `config.json` for the architecture and, when available,
 * `model.safetensors.index.json` for the exact checkpoint size
 * (metadata.total_size). Falls back to computing the parameter count
 * from the architecture dimensions (typically within ~2%).
 */

interface HFConfig {
  model_type?: string
  num_hidden_layers?: number
  hidden_size?: number
  num_attention_heads?: number
  num_key_value_heads?: number
  head_dim?: number
  vocab_size?: number
  intermediate_size?: number
  max_position_embeddings?: number
  tie_word_embeddings?: boolean
  torch_dtype?: string
  // MoE (Mixtral / Qwen / DeepSeek naming variants)
  num_local_experts?: number
  n_routed_experts?: number
  num_experts?: number
  num_experts_per_tok?: number
  moe_intermediate_size?: number
  // MLA (DeepSeek)
  kv_lora_rank?: number
  qk_rope_head_dim?: number
  // Some configs nest the text model (e.g. multimodal repos)
  text_config?: HFConfig
}

const DTYPE_BYTES: Record<string, number> = {
  float32: 4,
  bfloat16: 2,
  float16: 2,
  float8_e4m3fn: 1,
  float8_e5m2: 1,
}

export async function fetchHFModel(modelId: string): Promise<ModelArchitecture> {
  const id = modelId.trim().replace(/^https?:\/\/huggingface\.co\//, '').replace(/\/$/, '')
  const base = `https://huggingface.co/${id}/resolve/main`

  const configRes = await fetch(`${base}/config.json`)
  if (!configRes.ok) {
    throw new Error(
      configRes.status === 401 || configRes.status === 403
        ? `"${id}" is gated on Hugging Face — enter its architecture manually.`
        : `Could not fetch config.json for "${id}" (HTTP ${configRes.status}).`,
    )
  }
  let config = (await configRes.json()) as HFConfig
  if (!config.num_hidden_layers && config.text_config) config = config.text_config

  const arch = parseHFConfig(id, config)

  // Prefer the exact checkpoint size from the safetensors index.
  try {
    const idxRes = await fetch(`${base}/model.safetensors.index.json`)
    if (idxRes.ok) {
      const idx = (await idxRes.json()) as { metadata?: { total_size?: number } }
      const totalBytes = idx.metadata?.total_size
      const dtypeBytes = DTYPE_BYTES[config.torch_dtype ?? 'bfloat16'] ?? 2
      if (totalBytes && totalBytes > 0) {
        arch.paramsTotal = Math.round(totalBytes / dtypeBytes)
        if (!arch.moe) arch.paramsActive = arch.paramsTotal
      }
    }
  } catch {
    // single-file checkpoints have no index; the estimate stands
  }

  return arch
}

export function parseHFConfig(id: string, c: HFConfig): ModelArchitecture {
  const required = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'vocab_size'] as const
  for (const k of required) {
    if (typeof c[k] !== 'number') throw new Error(`config.json for "${id}" is missing "${k}" — is this a transformer LM?`)
  }
  const numLayers = c.num_hidden_layers!
  const hiddenSize = c.hidden_size!
  const heads = c.num_attention_heads!
  const kvHeads = c.num_key_value_heads ?? heads
  const headDim = c.head_dim ?? Math.floor(hiddenSize / heads)
  const vocabSize = c.vocab_size!
  const intermediateSize = c.intermediate_size ?? 4 * hiddenSize

  const numExperts = c.num_local_experts ?? c.n_routed_experts ?? c.num_experts
  const isMoE = typeof numExperts === 'number' && numExperts > 1
  const isMLA = typeof c.kv_lora_rank === 'number' && c.kv_lora_rank > 0

  const arch: ModelArchitecture = {
    name: id.split('/').pop() ?? id,
    paramsTotal: 0,
    paramsActive: 0,
    numLayers,
    hiddenSize,
    numAttentionHeads: heads,
    numKVHeads: kvHeads,
    headDim,
    vocabSize,
    intermediateSize,
    attentionType: isMLA ? 'mla' : kvHeads < heads ? 'gqa' : 'mha',
    maxContextLength: c.max_position_embeddings,
  }
  if (isMoE) {
    arch.moe = {
      numExperts: numExperts!,
      expertsPerToken: c.num_experts_per_tok ?? 2,
      expertIntermediateSize: c.moe_intermediate_size ?? intermediateSize,
    }
  }
  if (isMLA) {
    arch.mla = { kvLoraRank: c.kv_lora_rank!, qkRopeHeadDim: c.qk_rope_head_dim ?? 64 }
  }

  arch.paramsTotal = estimateParams(arch, c.tie_word_embeddings ?? false)
  arch.paramsActive = arch.moe
    ? estimateParams({ ...arch, moe: { ...arch.moe, numExperts: arch.moe.expertsPerToken } }, c.tie_word_embeddings ?? false)
    : arch.paramsTotal

  return arch
}

/**
 * Estimate parameter count from architecture dimensions, assuming a
 * standard pre-norm transformer with SwiGLU MLP (3 matrices).
 */
export function estimateParams(m: ModelArchitecture, tiedEmbeddings: boolean): number {
  const embed = m.vocabSize * m.hiddenSize * (tiedEmbeddings ? 1 : 2)
  const qOut = m.numAttentionHeads * m.headDim
  const kvOut = m.numKVHeads * m.headDim
  const attn = m.hiddenSize * qOut + 2 * m.hiddenSize * kvOut + qOut * m.hiddenSize
  const mlpPerUnit = (inter: number) => 3 * m.hiddenSize * inter
  const mlp = m.moe
    ? m.moe.numExperts * mlpPerUnit(m.moe.expertIntermediateSize) + m.hiddenSize * m.moe.numExperts // + router
    : mlpPerUnit(m.intermediateSize)
  const norms = 2 * m.hiddenSize
  return embed + m.numLayers * (attn + mlp + norms)
}
