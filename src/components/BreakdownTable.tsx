import type { MemoryEstimate } from '../engine/types'
import { fmtBytes } from '../engine/inference'
import { COMPONENT_COLORS } from './colors'

/**
 * Legend + data table for the gauge (also the accessible fallback for it).
 */
export function BreakdownTable({ estimate }: { estimate: MemoryEstimate }) {
  return (
    <div>
      <table className="breakdown">
        <thead>
          <tr>
            <th scope="col">COMPONENT</th>
            <th scope="col">PER GPU</th>
            <th scope="col">SHARE</th>
          </tr>
        </thead>
        <tbody>
          {estimate.components.map((c) => (
            <tr key={c.id}>
              <td>
                <span style={{ display: 'flex', alignItems: 'center' }}>
                  <span className="swatch" style={{ background: COMPONENT_COLORS[c.id] }} />
                  {c.label}
                </span>
                <span className="breakdown-detail">{c.detail}</span>
              </td>
              <td className="num">{fmtBytes(c.bytesPerGpu)}</td>
              <td className="num">{((c.bytesPerGpu / estimate.totalBytesPerGpu) * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <td>
              Total{estimate.gpusPerReplica > 1 ? ` — ${fmtBytes(estimate.totalBytes)} across ${estimate.gpusPerReplica} GPUs` : ''}
            </td>
            <td className="num">{fmtBytes(estimate.totalBytesPerGpu)}</td>
            <td className="num">100%</td>
          </tr>
        </tfoot>
      </table>
    </div>
  )
}
