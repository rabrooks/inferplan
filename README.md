# InferPlan — plan LLM deployments before you buy the GPUs

Interactive calculators for sizing LLM deployments: how much GPU memory a
model needs, which GPUs it fits on, and how far a given cluster can stretch.
All calculation runs client-side; the site is a static build with no backend.

**Scenario 1 (inference) is live.** The engine and UI are structured so the
remaining scenarios extend the same primitives rather than starting over.

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
| Training (optimizer states, ZeRO/FSDP, gradient checkpointing) | planned — next |
| Fine-tuning (full FT, LoRA, QLoRA) | planned |
| llm-d multi-pod capacity planning (disaggregated prefill/decode, request-rate-driven sizing) | planned |

Also planned: comparison mode (side-by-side configs), rough throughput and
latency estimates from the recorded bandwidth/FLOPS, cloud cost estimates,
and a custom-architecture editor.

## How the numbers are computed

See [docs/FORMULAS.md](docs/FORMULAS.md). Estimates target a vLLM-style
serving engine at 90% memory utilization; the activation peak is an explicit
heuristic. Treat totals as ±5% — this is a planning tool, not a benchmark.

## Contributing

Model presets live in `src/data/models.ts`, GPUs in `src/data/gpus.ts`, and
every formula in `src/engine/` has a corresponding test in
`src/engine/engine.test.ts`. PRs that add models/GPUs or tighten a formula
against measured numbers are very welcome.
