# Formulas

Inference estimates model a vLLM-style serving engine; training estimates
model a DeepSpeed/FSDP-style data-parallel run; llm-d capacity planning
models a disaggregated prefill/decode deployment. Symbols: `P` total params,
`P_act` active params per token, `L` layers, `H_kv` KV heads, `d_h` head
dim, `T` total tokens in KV cache (context length × concurrent sequences),
`tp`/`pp` tensor/pipeline degree, `N` data-parallel world size, `s` sequence
length, `b` micro-batch size, `h` hidden size, `a` attention heads.

## Weights

```
weights_bytes = P × bits_per_param / 8
per_gpu       = weights_bytes / (tp × pp)
```

`bits_per_param` is the *effective* storage cost, including quantization
metadata: group-quantized formats (group size 128) carry an FP16 scale and
often a zero-point per group, which is why INT4 is modeled as 4.5 bits and
INT8 (weight-only) as 8.25. GGUF k-quant values (Q4_K_M = 4.85, Q5_K_M = 5.69,
Q8_0 = 8.5) are the published effective bpw. For MoE models `P` is the total
count — every expert is resident even though only `k` are active per token.

## KV cache

Standard MHA/GQA:

```
kv_bytes_per_token = 2 × L × H_kv × d_h × kv_bits / 8
kv_total           = kv_bytes_per_token × T
```

MLA (DeepSeek): K and V share a compressed latent, so the cached elements per
layer are `kv_lora_rank + qk_rope_head_dim` (e.g. 512 + 64 for DeepSeek-V3),
replacing `2 × H_kv × d_h`.

Sharding: TP splits the cache across KV heads, so the per-GPU divisor is
`min(tp, H_kv)` — beyond that, engines replicate KV heads and the cache stops
shrinking (a warning is shown). The MLA latent is replicated across TP ranks
(as in vLLM); only PP shards it. PP divides the cache by `pp` (each stage
caches only its own layers).

## Activations (inference)

Serving engines don't retain per-layer activations; the transient peak is a
few hidden-state-sized buffers over the widest in-flight token batch:

```
act_bytes ≈ min(T, 8192) × hidden_size × 6 × act_bits / 8
```

8192 approximates the engine's max batched tokens per step; 6 covers
QKV/attention-out/MLP buffers; `act_bits` is 16 unless the format quantizes
activations (W8A8). This is an explicit heuristic — it is small relative to
weights and KV, and labelled an estimate in the UI.

## Training states

Persistent per-parameter bytes, following EleutherAI's transformer-math and
cross-checked against George614/gpu-mem-calculator:

| Precision | Weights | Gradients | Optimizer states |
|-----------|---------|-----------|------------------|
| Mixed BF16 | 2 | 2 | fp32 master 4 + optimizer state |
| FP32 | 4 | 4 | optimizer state only (weights are the master) |

Optimizer state per param: AdamW = 8 (fp32 m+v), AdamW 8-bit = 2
(bitsandbytes quantized m+v), SGD with momentum = 4. The mixed-precision
AdamW total is the standard **16 B/param** (2 + 2 + 4 + 8): an 8B-param model
carries ~128 GB of persistent state before activations.

ZeRO sharding across `N` data-parallel ranks (stage 3 ≈ FSDP full-shard):

```
stage 1: optimizer / N
stage 2: optimizer / N, gradients / N
stage 3: optimizer / N, gradients / N, weights / N
```

Stage 0 (plain DDP) replicates everything, so adding GPUs never reduces
per-GPU memory. Tensor/pipeline parallelism is not modeled for training yet.
MoE models are simplified to dense (all experts resident, no expert
parallelism) with a warning.

## Fine-tuning (LoRA / QLoRA)

Fine-tuning is a mode on the training estimator: full FT is the formula
above verbatim; LoRA/QLoRA change only *which* parameters carry gradients
and optimizer states.

Adapter parameters: each targeted projection `W ∈ R^(d_in×d_out)` gains
rank-`r` factors, summed over layers (GQA-aware — K/V project to
`H_kv × d_h`; MoE MLP targets count every expert):

```
adapter_params = r × Σ_targets (d_in + d_out) × L
```

Pinned to the LoRA paper (Hu et al. 2021): GPT-3 175B with r=4 on Q and V
gives ~18.9M adapter params — the claimed ~10,000× reduction in trainable
parameters (350 GB checkpoint → ~35 MB).

Per-GPU state with a frozen base (`A` = adapter params, `P` = base):

```
weights   = P × base_bytes + A × w_bytes     base_bytes: bf16 = 2,
gradients = A × g_bytes                        QLoRA NF4+DQ = 4.127 bits / 8
optimizer = A × opt_bytes
```

`w/g/opt_bytes` come from the training table above; ZeRO stage rules are
unchanged (FSDP shards frozen params too). QLoRA's 4.127 effective bits are
the paper's NF4 + double quantization: 4-bit weights, block-64 fp32 scales
double-quantized to fp8 with fp32 constants per 256 blocks —
`4 + 8/64 + 32/(64·256)`. Embeddings/lm_head actually stay 16-bit; treating
them as quantized underestimates by <2%. QLoRA also adds a transient
dequantization buffer to overhead (bitsandbytes dequantizes each NF4 tensor
to bf16 for its matmul): one bf16 copy of the largest linear layer. Paged
optimizer states are modeled as GPU-resident — paging is an OOM escape
valve, not a steady-state reduction.

Activations are **unchanged** from full training: the frozen base still
runs forward and backward, which is why activations dominate LoRA memory.

Pinned to the QLoRA paper (Dettmers et al. 2023): LLaMA 65B with r=64
adapters on all linear layers estimates to ~48 GB on one GPU — the paper's
"single 48GB GPU" claim — versus >780 GB for 16-bit full fine-tuning.

## Activations (training)

From Korthikanti et al., "Reducing Activation Recomputation in Large
Transformer Models" (also transformer-math), at TP = 1, bytes per GPU:

```
no recomputation:  s·b·h·L·(34 + 5·a·s/h)
selective:         s·b·h·L·34               (attention scores recomputed — what
                                             FlashAttention avoids storing)
full:              s·b·h·L·2                (only layer inputs retained)
```

Constants assume 2-byte activations (fp32 training doubles them) and a
standard 4h MLP — SwiGLU models run slightly higher, and embedding/logit
buffers are excluded, so training totals are labelled ±10%. The inverse
solver subtracts the persistent states from usable VRAM and solves the
remaining headroom for max micro-batch (linear) and max sequence length
(binary search — quadratic in `s` without recomputation).

## Overhead and fit

Each GPU pays ~0.75 GiB for CUDA context, NCCL buffers, and allocator
bookkeeping. Fit checks compare against `VRAM × 0.9` (the engine's memory
utilization fraction, leaving headroom for fragmentation).

## Inverse solver

```
free_for_kv = VRAM × 0.9 − weights/gpu − activations/gpu − overhead
max_tokens  = free_for_kv / kv_bytes_per_token_per_gpu
```

Max context at a concurrency and max concurrency at a context length both
derive from `max_tokens`. The "smallest deployment" search tries power-of-two
GPU counts, preferring TP while attention heads divide evenly and pushing the
remainder into PP.

## llm-d capacity planning (disaggregated prefill/decode)

Answers "how many GPUs does this request rate need, within these latency
SLOs?" for a deployment split into a compute-bound **prefill pool** and a
bandwidth-bound **decode pool** with KV transfer between them (the
llm-d / [DistServe][distserve] / [Splitwise][splitwise] pattern). The model
is a first-principles roofline that doubles as the latency model: the same
physics yields throughput, TTFT, and TPOT for every GPU in the database
from its spec-sheet bandwidth and TFLOPS. Workload symbols: `λ` requests/s,
`T_in`/`T_out` mean input/output tokens per request.

Every efficiency assumption is an explicit knob, not a baked-in constant,
so a measured trace (e.g. from [InferLens](interop.md)) can replace each
default with a value calibrated to your hardware.

### Prefill pool — compute-bound, sized to the TTFT SLO

```
FLOPs/request = 2·P_act·T_in  +  2·T_in²·(a·d_h)·L
t_prefill     = FLOPs / (pod_gpus × TFLOPS_bf16 × MFU)        MFU default 0.45
TTFT          = W_queue + β·t_prefill                         β default 1.8
```

The `2·P_act·T` GEMM term is the Kaplan 2·FLOPs-per-weight rule (active
params for MoE). The quadratic causal-attention term is **always included**
— negligible at chat context, it is ~84% of total FLOPs for Llama3-405B at
1M tokens, and it costs nothing to compute, so no crossover heuristic is
needed. Pinned against Meta's measured 405B prefills (128K tokens ≈ 60 s on
one 8×H100 host; 1M ≈ 77 s at their reported 63% utilization) from the
context-parallelism paper ([arXiv:2411.01783][meta-cp]). MFU defaults to
0.45 — the 0.4–0.5 band multiple sources report for dense H100-class
prefill; 0.6+ appears only in attention-dominated extreme-long-context runs.

`β` is the KV-transfer overhead of shipping the prefilled cache to the
decode pool, published at 1.8–1.9× raw prefill time and strongly
interconnect-dependent — a documented heuristic knob.

`W_queue` is the Erlang C (M/M/c) mean wait with the pool's pods as
servers: offered load `a = λ·t_prefill`, wait probability from the Erlang B
recursion, `W_q = C(c,a)·t_prefill/(c−a)`. A service-time-only TTFT would
be silently wrong at high utilization — queueing is not optional. The pool
is the smallest pod count with utilization `ρ = a/c ≤ 0.85` (the queueing
stability cap used across the field) *and* `TTFT ≤ SLO`. Mean wait is what
ships today; a two-moment P99 approximation is on the backlog.

### Decode pool — bandwidth-bound, sized to the TPOT SLO

Each decode step streams the weight shard once per GPU plus every running
sequence's KV shard; at batch `n` with mean cached context
`T̄ = T_in + T_out/2` (requests in a continuously batched pod are uniformly
spread through their generations):

```
t_step(n) = (weights_read(n)/gpu + n × kv_bytes(T̄)/gpu) / (BW × eff)
TPOT      = t_step(n)                                    eff default 0.7
```

`eff` is the fraction of spec-sheet HBM bandwidth realized in practice
(band 0.5–0.8). Weight traffic amortizes across the batch; KV traffic does
not — attention stays memory-bound at any batch size
([arXiv:2503.08311][mind-gap]). At n=1 with no KV this reduces to the
familiar ceiling `tok/s = BW / weight_bytes`, pinned to the [BentoML
handbook][bentoml]'s worked example (70B FP16 ≈ 140 GB on a 3.35 TB/s H100
→ ~24 tok/s per sequence).

`weights_read(n)`: dense models stream all weights every step. MoE models
stream only the experts the batch routes to — expected distinct experts
for `n` tokens choosing `k` of `E` is `E·(1−(1−k/E)^n)`, rescaled so batch
1 reads exactly the active parameters and a full batch approaches the
total. This is why batched MoE decode approaches dense-total bandwidth
cost while batch-1 MoE decode is dramatically faster.

Per-pod concurrency is capped by the TPOT SLO (largest `n` with
`t_step(n) ≤ SLO`) **and** by KV capacity (the memory inverse solver at
mean context `T̄`) — the UI reports which bound bites. The pod count is the
smallest `p` whose steady-state batch — the Little's law fixed point
`n = (λ/p)·T_out·t_step(n)` — stays under 0.85 of that cap. A warning
flags very large batches where decode approaches the compute roofline and
the bandwidth-only estimate turns optimistic.

### Cross-checks and outputs

The linear `t_step(n)` is the same shape as inference-fleet-sim's
calibrated `t_iter(n) = W + H·n` ([arXiv:2603.16054][fleet-sim]); our
physically derived W and H land within the same order of magnitude as
their hand-calibrated H100 constants (derived H ≈ 0.8 ms/slot vs. their
0.32 — their constants bake in unstated TP/quantization assumptions), and
that agreement is enforced as a loose test, not imported as constants.
Sizing to explicit TTFT/TPOT SLOs with a hard ρ cap follows the goodput
methodology of [DistServe][distserve] and [arXiv:2508.01989][pd-slo].

All predictions are emitted in the units stock vLLM exposes and InferLens
records — predicted `num_running_reqs` per decode pod, `kv_cache_usage`
fraction, queued/prefill/decode phase times — per [docs/interop.md](interop.md).

Caveats: mean token counts stand in for full arrival/length distributions;
M/M/c assumes exponential-ish service times (mean wait only); pod counts
are a starting point for load testing, not a guarantee.

[distserve]: https://arxiv.org/abs/2401.09670
[splitwise]: https://arxiv.org/abs/2311.18677
[meta-cp]: https://arxiv.org/abs/2411.01783
[mind-gap]: https://arxiv.org/abs/2503.08311
[fleet-sim]: https://arxiv.org/abs/2603.16054
[pd-slo]: https://arxiv.org/abs/2508.01989
[bentoml]: https://bentoml.com/llm/getting-started/choosing-the-right-gpu

## Parameter estimation (HF import)

When a repo has `model.safetensors.index.json`, the exact checkpoint size
(`metadata.total_size ÷ dtype bytes`) is used. Otherwise parameters are
computed from the config assuming a pre-norm transformer with SwiGLU MLP:

```
P ≈ vocab × hidden × (1 or 2)                            # embeddings (tied?)
  + L × [ hidden×(q + out) + 2×hidden×kv + 3×hidden×I ]  # attention + MLP
```

with the MLP term multiplied out per expert for MoE. Verified within ~2–3% of
published counts for Llama/Qwen/Mistral in the test suite.
