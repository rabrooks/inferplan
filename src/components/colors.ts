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
  gradients: '#9085e9',
  optimizer: '#d55181',
  overhead: 'var(--c-ovh)',
}
