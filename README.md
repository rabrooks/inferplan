# InferPlan — plan LLM deployments before you buy the GPUs

Interactive calculators for sizing LLM deployments: how much GPU memory a
model needs, which GPUs it fits on, and how far a given cluster can stretch.
All calculation runs client-side; the site is a static build with no backend.

**Inference, training, and fine-tuning scenarios are live.** The engine and UI are
structured so the remaining scenarios extend the same primitives rather than
starting over.

Sister project: [InferLens](https://github.com/rabrooks/inferlens) —
engine-neutral observability for inference engines. **InferPlan predicts what
a deployment needs; InferLens shows what it actually did.**

## Features

- **Memory breakdown** — weights, KV cache, activation peak, and framework
  overhead, per GPU, rendered as a VRAM gauge against the selected GPU's
  capacity.
- **Model presets + Hugging Face import** — presets for Llama, Qwen, Mistral,
  Mixtral, DeepSeek, and GPT-OSS, or paste any HF model id to import its
  `config.json`. Exact parameter counts come from the safetensors index when
  available; otherwise they are derived from the architecture (±2%).
- **MoE-aware** — total vs. active parameters are tracked separately, so
  Mixtral/DeepSeek/Qwen-MoE weight memory is computed from all experts.
- **MLA-aware** — DeepSeek-style latent attention uses the compressed KV
  formula, not the GQA one.
- **Quantization** — weight formats from FP32 down to INT3 and GGUF k-quants
  (effective bits include scale/zero-point metadata), with KV-cache precision
  as an independent choice (FP8 KV halves the biggest scaling term).
- **Multi-GPU** — tensor and pipeline parallelism with the real sharding
  rules: head-divisibility checks, KV replication when TP exceeds KV heads.
- **GPU database** — B200 through RTX 4090 and MI300X/MI325X, with memory
  bandwidth and FLOPS recorded for the throughput estimates coming in later
  phases.
- **Inverse solver** — max context length / max concurrent sequences for a
  given GPU, and the smallest count of every GPU model that serves the config.
- **Training memory** — mixed-precision or FP32 states (AdamW / AdamW 8-bit /
  SGD), ZeRO stages 1–3 (≈ FSDP full-shard at 3) sharded across data-parallel
  ranks, and the Korthikanti et al. activation formula with
  none/selective/full checkpointing. The same inverse solver reports
  activation headroom, max micro-batch, and max sequence length.
- **Fine-tuning modes** — full FT, LoRA, and QLoRA on the same training
  engine: frozen bf16 or NF4 (4-bit, double-quantized) base weights,
  adapter-only gradients/optimizer states from rank × targeted projections,
  and the QLoRA dequant buffer — pinned to the LoRA and QLoRA papers'
  published numbers (65B on a single 48 GB GPU).
- **Shareable URLs** — the entire configuration (including imported custom
  models) is encoded in query params, so a config can be linked in an issue
  or a Slack thread.

## Quick start

```sh
npm install
npm run dev     # local dev server
npm test        # engine unit tests (vitest)
npm run build   # static production build in dist/
```

Stack: Vite + React + TypeScript + Tailwind. The calculation engine
(`src/engine/`) is pure TypeScript with no React imports — it is unit-tested
independently and reusable outside the UI.

## Roadmap

| Scenario | Status |
|----------|--------|
| Inference, single GPU | ✅ shipped |
| Inference, multi-GPU (TP/PP) | ✅ shipped |
| Training (optimizer states, ZeRO/FSDP, activation checkpointing) | ✅ shipped |
| Fine-tuning (full FT, LoRA, QLoRA) | ✅ shipped |
| llm-d multi-pod capacity planning (disaggregated prefill/decode, request-rate-driven sizing) | planned — next |

Also planned: comparison mode (side-by-side configs), rough throughput and
latency estimates from the recorded bandwidth/FLOPS, cloud cost estimates,
and a custom-architecture editor.

## How InferPlan differs

Plenty of VRAM calculators exist; most answer "can I load it," not "can I
serve it" (Hugging Face's own `accelerate estimate-memory` is weights-only).
The polished ones use fixed model dropdowns, skip MoE/MLA, and apply no real
sharding rules. InferPlan's lane:

- **Architecture fidelity** — exact params from the safetensors index, MoE
  total-vs-active accounting, DeepSeek-style MLA latent KV, and the
  KV-replication rule when TP exceeds KV heads.
- **Training and inference in one engine** — the same audited primitives
  compute serving KV cache and ZeRO-sharded optimizer states; no other tool
  unifies the two (training-only prior art:
  [gpu-mem-calculator](https://github.com/George614/gpu-mem-calculator)).
- **Inverse solvers, not just totals** — max context/concurrency for
  inference, max micro-batch/sequence length for training, and the smallest
  deployment of every GPU in the database.
- **An auditable engine** — pure TypeScript, no backend, every formula pinned
  by a test against published numbers ([docs/FORMULAS.md](docs/FORMULAS.md)).
- **Fleet sizing is coming** — disaggregated prefill/decode capacity planning
  (the llm-d scenario) has essentially no tooling today; that's the road this
  engine is built for.

## How the numbers are computed

See [docs/FORMULAS.md](docs/FORMULAS.md). Inference estimates target a
vLLM-style serving engine at 90% memory utilization; training estimates
follow transformer-math / Korthikanti et al. Activation peaks are explicit
heuristics — treat totals as ±5% (inference) / ±10% (training). This is a
planning tool, not a benchmark.

## Contributing

Model presets live in `src/data/models.ts`, GPUs in `src/data/gpus.ts`, and
every formula in `src/engine/` has a corresponding test in
`src/engine/engine.test.ts`. PRs that add models/GPUs or tighten a formula
against measured numbers are very welcome.

## License

[Apache-2.0](LICENSE)
