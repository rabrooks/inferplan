import { minGpusToFit } from '../engine/fit'
import type { InferenceWorkload, ModelArchitecture, QuantizationConfig } from '../engine/types'
import { GPU_DATABASE } from '../data/gpus'

interface Props {
  model: ModelArchitecture
  quant: QuantizationConfig
  workload: InferenceWorkload
  selectedGpuId: string
  onSelect: (gpuId: string) => void
}

/**
 * One card per GPU in the database: the smallest count of that GPU that
 * serves the current configuration. Clicking a card selects it as the
 * gauge/solver target.
 */
export function FitGrid({ model, quant, workload, selectedGpuId, onSelect }: Props) {
  return (
    <div className="fit-grid">
      {GPU_DATABASE.map((gpu) => {
        const fit = minGpusToFit(model, quant, workload, gpu)
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
                {fit.count}× <span className="fit-card-shape">
                  {fit.count > 1
                    ? `TP ${fit.shape.tensorParallel}${fit.shape.pipelineParallel > 1 ? ` · PP ${fit.shape.pipelineParallel}` : ''}`
                    : 'single GPU'}
                </span>
              </div>
            ) : (
              <div className="fit-card-verdict no">&gt;128 GPUs</div>
            )}
          </button>
        )
      })}
    </div>
  )
}
