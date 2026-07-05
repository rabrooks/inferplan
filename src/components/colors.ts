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
