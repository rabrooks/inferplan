import type { GPUSpec } from '../engine/types'

/**
 * GPU database. bf16TFLOPS is dense (non-sparse) tensor-core throughput;
 * bandwidth and FLOPS feed the llm-d throughput/latency roofline (llmd.ts).
 */
export const GPU_DATABASE: GPUSpec[] = [
  { id: 'b200', name: 'NVIDIA B200', vendor: 'nvidia', vramGiB: 192, memoryBandwidthGBs: 8000, bf16TFLOPS: 2250 },
  { id: 'h200', name: 'NVIDIA H200 (SXM)', vendor: 'nvidia', vramGiB: 141, memoryBandwidthGBs: 4800, bf16TFLOPS: 989 },
  { id: 'h100-sxm', name: 'NVIDIA H100 (SXM)', vendor: 'nvidia', vramGiB: 80, memoryBandwidthGBs: 3350, bf16TFLOPS: 989 },
  { id: 'h100-nvl', name: 'NVIDIA H100 NVL', vendor: 'nvidia', vramGiB: 94, memoryBandwidthGBs: 3900, bf16TFLOPS: 835 },
  { id: 'a100-80', name: 'NVIDIA A100 80GB', vendor: 'nvidia', vramGiB: 80, memoryBandwidthGBs: 2039, bf16TFLOPS: 312 },
  { id: 'a100-40', name: 'NVIDIA A100 40GB', vendor: 'nvidia', vramGiB: 40, memoryBandwidthGBs: 1555, bf16TFLOPS: 312 },
  { id: 'l40s', name: 'NVIDIA L40S', vendor: 'nvidia', vramGiB: 48, memoryBandwidthGBs: 864, bf16TFLOPS: 362 },
  { id: 'l4', name: 'NVIDIA L4', vendor: 'nvidia', vramGiB: 24, memoryBandwidthGBs: 300, bf16TFLOPS: 121 },
  { id: 'a10g', name: 'NVIDIA A10G', vendor: 'nvidia', vramGiB: 24, memoryBandwidthGBs: 600, bf16TFLOPS: 70 },
  { id: 'rtx5090', name: 'NVIDIA RTX 5090', vendor: 'nvidia', vramGiB: 32, memoryBandwidthGBs: 1792, bf16TFLOPS: 210 },
  { id: 'rtx4090', name: 'NVIDIA RTX 4090', vendor: 'nvidia', vramGiB: 24, memoryBandwidthGBs: 1008, bf16TFLOPS: 165 },
  { id: 'mi300x', name: 'AMD MI300X', vendor: 'amd', vramGiB: 192, memoryBandwidthGBs: 5300, bf16TFLOPS: 1307 },
  { id: 'mi325x', name: 'AMD MI325X', vendor: 'amd', vramGiB: 256, memoryBandwidthGBs: 6000, bf16TFLOPS: 1307 },
]
