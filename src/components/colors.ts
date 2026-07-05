import type { MemoryComponent } from '../engine/types'

/**
 * Fixed component→color assignment, shared by the gauge and the table.
 * Training-phase components already have reserved slots so colors stay
 * stable across scenarios.
 */
export const COMPONENT_COLORS: Record<MemoryComponent['id'], string> = {
  weights: 'var(--c-weights)',
  kvCache: 'var(--c-kv)',
  activations: 'var(--c-act)',
  gradients: 'var(--c-grad)',
  optimizer: 'var(--c-opt)',
  overhead: 'var(--c-ovh)',
}

/**
 * Ink for the gauge's on-stratum labels. Most strata are light enough in
 * both themes for the shared near-black ink; the gradients stratum flips
 * per theme (light-theme violet #4a3aa7 is dark — black ink is ~2.4:1).
 */
export const COMPONENT_LABEL_INK: Record<MemoryComponent['id'], string> = {
  weights: 'var(--gauge-label-ink)',
  kvCache: 'var(--gauge-label-ink)',
  activations: 'var(--gauge-label-ink)',
  gradients: 'var(--gauge-label-ink-grad)',
  optimizer: 'var(--gauge-label-ink-opt)',
  overhead: 'var(--gauge-label-ink)',
}
