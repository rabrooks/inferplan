# InferPlan ↔ InferLens: how the two projects fit together

> **This document is mirrored in both repositories** —
> [`inferplan/docs/interop.md`](https://github.com/rabrooks/inferplan/blob/main/docs/interop.md)
> and
> [`inferlens/docs/interop.md`](https://github.com/rabrooks/inferlens/blob/main/docs/interop.md).
> It is the shared contract between the two projects: if a change to either
> project affects anything described here, update **both copies** in the same
> spirit as a schema change. Keeping them identical is a convention, not
> tooling — reviewers should check.

**[InferPlan](https://github.com/rabrooks/inferplan)** predicts what an LLM
deployment will need — GPU memory, GPU counts, throughput, latency — from
published formulas and hardware spec sheets, *before* anything is deployed.
It is a static calculator: pure TypeScript engine, every formula cited and
unit-tested against published or measured numbers
([`docs/FORMULAS.md`](https://github.com/rabrooks/inferplan/blob/main/docs/FORMULAS.md)).

**[InferLens](https://github.com/rabrooks/inferlens)** records what an
inference engine *actually did* — per-step scheduling, batching, KV-cache
behavior, per-request lifecycle timing — as a portable, replayable trace
captured from a stock vLLM build (no fork, no monkeypatching).

One sentence for both: **InferPlan predicts, InferLens observes.**

## The loop

```text
        ┌─────────────────────────────────────────────────────┐
        │                                                     │
        ▼                                                     │
   InferPlan ──── plan ────► deploy ──── run ────► InferLens ─┘
   (predicted n, KV%,                     (measured n, KV%,
    phase timings)                         phase timings)
```

1. **Plan.** InferPlan sizes the deployment: how many GPUs, what
   concurrency, what latency to expect.
2. **Deploy and observe.** InferLens records a trace of the live engine.
3. **Compare.** Because both sides speak the same vocabulary (below), a
   trace can be compared against the prediction number-for-number.
4. **Re-plan.** Where they disagree, the trace calibrates InferPlan's
   explicit heuristic knobs — turning assumed efficiency factors into
   measured ones for *your* hardware and workload.

Neither project depends on the other to be useful. InferPlan works with no
trace; InferLens works with no plan. The loop is what they enable together.

## Where they meet: disaggregated serving (llm-d)

Disaggregated serving (as in [llm-d](https://llm-d.ai)) splits a deployment
into a compute-bound **prefill pool** and a memory-bandwidth-bound
**decode pool**, with KV-cache transfer between them.

- InferPlan's fleet-sizing scenario answers "how many prefill pods and how
  many decode pods does this request rate need, within these latency
  targets?" — *before* deploy.
- InferLens's trace model answers "what did the scheduler and KV cache in
  each pool actually do?" — *after* deploy. (Each pool is recorded as an
  ordinary single-engine trace today; correlating traces *across* pools is
  future InferLens work — see the honesty table below.)

## The shared vocabulary (the interface)

The contract: **InferPlan emits predictions in exactly the units InferLens
measures**, and InferLens's units are the ones stock vLLM exposes. No unit
conversion, no reinterpretation — a trace should overlay a prediction
directly.

| Quantity | InferPlan predicts | InferLens measures (vLLM source) |
|---|---|---|
| Batch concurrency | steady-state running requests per decode pod, `n` | `SchedulerStats.num_running_reqs` per engine |
| KV-cache pressure | KV usage as a fraction of KV capacity (0–1) | `SchedulerStats.kv_cache_usage` per engine |
| Request phase timing | queued / prefill (TTFT) / decode (TPOT) time distributions | `FinishedRequestStats` per-request queued/prefill/decode times |
| Throughput | tokens/s per pod and per pool | tokens/s aggregated from per-request token counts and timings |

Rules that keep this contract honest:

- **InferPlan side:** any new predicted quantity should name the vLLM/
  InferLens counter it corresponds to, or explicitly state that no
  observable counterpart exists yet. Predictions that can never be checked
  are how calculators drift into fiction.
- **InferLens side:** the trace schema
  ([`TRACE_SPEC.md`](https://github.com/rabrooks/inferlens/blob/main/docs/TRACE_SPEC.md))
  should preserve these quantities losslessly from the engine's stats. If a
  schema change renames or rescales one of them, that is an interface
  change — update this document and flag it in the PR.

## Calibration: the knobs a trace can measure

InferPlan deliberately keeps its efficiency heuristics as **explicit,
documented knobs** rather than baked-in constants, precisely so that a
trace can replace the default with a measured value:

| Knob | Default (documented assumption) | How a trace calibrates it |
|---|---|---|
| Prefill MFU (model FLOPs utilization) | ~0.4–0.5 for dense prefill on modern GPUs | measured prefill time per request vs. the FLOPs the model requires |
| Decode roofline efficiency | fraction of spec-sheet HBM bandwidth actually achieved | measured decode tokens/s vs. bytes-per-token × spec bandwidth |
| KV-transfer overhead (disaggregated) | ~1.8× raw prefill time (published range 1.8–1.9×) | not yet measurable — see below |

This is the design reason InferPlan uses first-principles physics
(FLOPs/MFU rooflines over hardware specs) instead of lookup tables of
profiled numbers: you can only calibrate a model that has explicit physical
knobs.

## What works today vs. what's gated (honesty table)

| Capability | Status | Gated on |
|---|---|---|
| Compare predicted vs. measured concurrency, KV fraction, phase timings for a **single pool** | ✅ works with stock vLLM | — |
| Record each pool of a disaggregated deployment as its own trace | ✅ works (one trace per engine) | — |
| **Correlated cross-pool** trace (one timeline spanning prefill + decode pools) | ❌ not yet | InferLens multi-node recording (designed for, not shipped) |
| **Step-granular** overlay (predicted per-step iteration time vs. measured per-step batch composition) | ❌ not yet | vLLM doesn't expose per-step batch composition to stat loggers — see [`upstream-gaps.md`](https://github.com/rabrooks/inferlens/blob/main/docs/upstream-gaps.md) gap 1 |
| Calibrate the KV-transfer overhead knob from a trace | ❌ not yet | no KV-transfer timing capture exists yet in the trace |

A useful side effect of the split pools: vLLM's missing per-step
prefill/decode token split (upstream gap 1) is unambiguous at the pod level
in a disaggregated deployment — a prefill pod's steps are all prefill, a
decode pod's are all decode. Aggregate-level comparison per pool works
today without waiting on upstream changes.

The upstream-gaps entries in InferLens are written as future vLLM RFC
candidates. InferPlan is a second motivating consumer for several of them:
exposing per-step scheduling detail is what would let a *planner's*
predictions be verified step-by-step, not just on averages. Contributors
filing those RFCs should cite both projects.

## For contributors

- Changes that touch this interface: prediction output types in InferPlan
  (`src/engine/`), trace schema or collector fields in InferLens, or the
  vocabulary table above. When in doubt, link this doc in your PR and say
  which row you're touching.
- Both projects are Apache-2.0. InferLens is pre-alpha and its trace
  schema is explicitly unstable pre-1.0; treat the vocabulary table's
  *names and units* as the stable part of the contract, not any specific
  serialization.
- Good first contribution to the loop itself: record an InferLens trace of
  a deployment you sized with InferPlan and report where prediction and
  measurement diverge — divergence reports are how the default knob values
  get better.
