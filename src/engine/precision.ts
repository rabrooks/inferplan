import type { PrecisionFormat } from './types'

/**
 * Weight-storage formats. bitsPerParam is effective bits including
 * quantization metadata: group-quantized formats (group size 128) carry
 * one FP16 scale (and often a zero-point) per group, adding ~0.25–0.5 bits.
 */
export const WEIGHT_FORMATS: PrecisionFormat[] = [
  { id: 'fp32', label: 'FP32', bitsPerParam: 32 },
  { id: 'bf16', label: 'BF16', bitsPerParam: 16 },
  { id: 'fp16', label: 'FP16', bitsPerParam: 16 },
  { id: 'fp8', label: 'FP8 (W8A16)', bitsPerParam: 8, note: 'FP8 weights, 16-bit activations' },
  { id: 'int8', label: 'INT8 (W8A16)', bitsPerParam: 8.25, note: 'weight-only, group-128 scales' },
  { id: 'w8a8', label: 'W8A8', bitsPerParam: 8.25, note: 'INT8/FP8 weights and activations', activationBits: 8 },
  { id: 'int4', label: 'INT4 (GPTQ/AWQ)', bitsPerParam: 4.5, note: 'group-128 scales + zero-points' },
  { id: 'nf4', label: 'NF4 (bitsandbytes)', bitsPerParam: 4.5, note: 'double quantization' },
  { id: 'int3', label: 'INT3', bitsPerParam: 3.5, note: 'group-128 scales' },
  { id: 'q4km', label: 'GGUF Q4_K_M', bitsPerParam: 4.85 },
  { id: 'q5km', label: 'GGUF Q5_K_M', bitsPerParam: 5.69 },
  { id: 'q8_0', label: 'GGUF Q8_0', bitsPerParam: 8.5 },
]

/** KV-cache element formats (independent of weight quantization). */
export const KV_FORMATS: PrecisionFormat[] = [
  { id: 'fp16', label: 'FP16 / BF16', bitsPerParam: 16 },
  { id: 'fp8', label: 'FP8', bitsPerParam: 8 },
  { id: 'int8', label: 'INT8', bitsPerParam: 8 },
  { id: 'int4', label: 'INT4', bitsPerParam: 4 },
]

export function weightFormat(id: string): PrecisionFormat {
  const f = WEIGHT_FORMATS.find((f) => f.id === id)
  if (!f) throw new Error(`Unknown weight format: ${id}`)
  return f
}

export function kvFormat(id: string): PrecisionFormat {
  const f = KV_FORMATS.find((f) => f.id === id)
  if (!f) throw new Error(`Unknown KV format: ${id}`)
  return f
}
