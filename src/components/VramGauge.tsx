import { useState } from 'react'
import type { MemoryComponent } from '../engine/types'
import { GiB, fmtBytes } from '../engine/inference'
import { COMPONENT_COLORS } from './colors'

interface Props {
  components: MemoryComponent[]
  capacityGiB: number
  memoryUtilization: number
}

/** Pick a tick step so the scale shows ~6–12 labels. */
function tickStep(scaleGiB: number): number {
  for (const step of [4, 8, 16, 32, 64, 128, 256]) {
    if (scaleGiB / step <= 12) return step
  }
  return 512
}

interface Hover {
  comp: MemoryComponent
  x: number
  y: number
}

/**
 * The VRAM gauge: the selected GPU rendered as a measuring vessel,
 * filling bottom-up with memory strata. The dashed line is the usable
 * limit (capacity × utilization); anything above it is hatched overflow.
 */
export function VramGauge({ components, capacityGiB, memoryUtilization }: Props) {
  const [hover, setHover] = useState<Hover | null>(null)

  const totalGiB = components.reduce((s, c) => s + c.bytesPerGpu, 0) / GiB
  const usableGiB = capacityGiB * memoryUtilization
  const scaleGiB = Math.max(capacityGiB, totalGiB) * 1.04
  const pct = (gib: number) => (gib / scaleGiB) * 100

  const step = tickStep(scaleGiB)
  const ticks: number[] = []
  for (let t = step; t <= scaleGiB; t += step) {
    // suppress ticks that would collide with the capacity labels
    if (Math.abs(t - usableGiB) > scaleGiB * 0.035 && Math.abs(t - capacityGiB) > scaleGiB * 0.035) {
      ticks.push(t)
    }
  }

  let cum = 0
  const strata = components.map((c) => {
    const startGiB = cum
    cum += c.bytesPerGpu / GiB
    return { comp: c, bottom: pct(startGiB), height: pct(c.bytesPerGpu / GiB) }
  })

  return (
    <div className="gauge-wrap" style={{ position: 'relative' }}>
      <div className="gauge-vessel" aria-hidden="true">
        {strata.map(({ comp, bottom, height }) => (
          <div
            key={comp.id}
            className="gauge-stratum"
            style={{
              bottom: `${bottom}%`,
              height: `max(calc(${height}% - 2px), 0px)`,
              background: COMPONENT_COLORS[comp.id],
            }}
            onMouseMove={(e) => {
              const rect = e.currentTarget.closest('.gauge-wrap')!.getBoundingClientRect()
              setHover({ comp, x: e.clientX - rect.left + 12, y: e.clientY - rect.top })
            }}
            onMouseLeave={() => setHover(null)}
          >
            {height > 5.5 && (
              <span className="gauge-stratum-label">
                {comp.label} · {fmtBytes(comp.bytesPerGpu)}
              </span>
            )}
          </div>
        ))}
        <div className="gauge-capacity-line" style={{ bottom: `${pct(usableGiB)}%` }} />
        {totalGiB > usableGiB && (
          <div
            className="gauge-overflow-veil"
            style={{ bottom: `${pct(usableGiB)}%`, height: `${pct(totalGiB - usableGiB)}%` }}
          />
        )}
      </div>
      <div className="gauge-scale" aria-hidden="true">
        {ticks.map((t) => (
          <div key={t} className="gauge-tick" style={{ bottom: `${pct(t)}%` }}>
            {t}
          </div>
        ))}
        <div className="gauge-cap-label" style={{ bottom: `${pct(usableGiB)}%` }}>
          {usableGiB.toFixed(0)} usable
        </div>
        <div className="gauge-cap-label" style={{ bottom: `${pct(capacityGiB)}%` }}>
          {capacityGiB} GiB
        </div>
      </div>
      {hover && (
        <div className="gauge-tooltip" style={{ left: hover.x, top: hover.y }}>
          <div className="tt-title">{hover.comp.label}</div>
          <div className="tt-val">{fmtBytes(hover.comp.bytesPerGpu)} per GPU</div>
          <div>{hover.comp.detail}</div>
        </div>
      )}
    </div>
  )
}
