import type { GPUSpec } from '../engine/types'
import { GPU_DATABASE } from '../data/gpus'

export interface FitCard {
  count: number
  /** e.g. "TP 4 · PP 2", "ZeRO-3 · DP 8", or "single GPU". */
  shapeLabel: string
}

interface Props {
  /** Smallest deployment of this GPU that holds the current config, or null. */
  fitFor: (gpu: GPUSpec) => FitCard | null
  /** Shown when fitFor returns null, e.g. ">128 GPUs". */
  noFitLabel: string
  selectedGpuId: string
  onSelect: (gpuId: string) => void
}

/**
 * One card per GPU in the database: the smallest count of that GPU that
 * holds the current configuration (scenario-specific search supplied by
 * the caller). Clicking a card selects it as the gauge/solver target.
 */
export function FitGrid({ fitFor, noFitLabel, selectedGpuId, onSelect }: Props) {
  return (
    <div className="fit-grid">
      {GPU_DATABASE.map((gpu) => {
        const fit = fitFor(gpu)
        return (
          <button
            key={gpu.id}
            className="fit-card"
            aria-pressed={gpu.id === selectedGpuId}
            onClick={() => onSelect(gpu.id)}
          >
            <div className="fit-card-name">{gpu.name}</div>
            <div className="fit-card-vram">
              {gpu.vramGiB} GiB · {gpu.memoryBandwidthGBs} GB/s
            </div>
            {fit ? (
              <div className="fit-card-verdict ok">
                {fit.count}× <span className="fit-card-shape">{fit.shapeLabel}</span>
              </div>
            ) : (
              <div className="fit-card-verdict no">&gt;{noFitLabel}</div>
            )}
          </button>
        )
      })}
    </div>
  )
}
