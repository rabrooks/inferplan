# Memory formulas

Inference estimates model a vLLM-style serving engine; training estimates
model a DeepSpeed/FSDP-style data-parallel run. Symbols: `P` total params,
`L` layers, `H_kv` KV heads, `d_h` head dim, `T` total tokens in KV cache
(context length × concurrent sequences), `tp`/`pp` tensor/pipeline degree,
`N` data-parallel world size, `s` sequence length, `b` micro-batch size,
`h` hidden size, `a` attention heads.

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
