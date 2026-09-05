# context-cuda

A C++/CUDA PyTorch extension for long-context Transformer inference.

The goal is to implement the inference-time attention path myself instead of
relying on PyTorch's built-in ops:

- **Custom attention kernels** — fused scaled dot-product attention with causal masking
- **RoPE** — rotary position embeddings applied directly in the kernel
- **KV-cache management** — allocate, append to, and read from a key/value cache
  so each new token doesn't recompute the whole sequence

Python is used for correctness tests against PyTorch reference implementations
and for benchmarking.

## Status

Early. Right now the repo contains the math groundwork:

- [attention-playground.py](attention-playground.py) — a heavily commented walkthrough of
  attention in plain PyTorch: Q/K/V shapes, the `[B, H, T, T]` score matrix,
  `1/sqrt(D)` scaling, causal masking with `-inf`, and softmax.

The CUDA kernels are not written yet.

## Concepts / notation

| Symbol | Meaning |
| --- | --- |
| `B` | batch size — how many sequences at once |
| `H` | number of attention heads |
| `T` | sequence length in tokens |
| `D` | head dimension — numbers per token, per head |

Q, K, and V are each `[B, H, T, D]`. Attention scores are `[B, H, T, T]`:
every query token scored against every key token.

## Requirements

- Python 3.10+
- PyTorch with CUDA
- CUDA toolkit (`nvcc`) matching the PyTorch build

## Running

```bash
python attention-playground.py
```

Prints tensor shapes and intermediate values at each step of attention.

## Roadmap

- [x] Attention math in PyTorch (scores, scaling, causal mask, softmax)
- [ ] Attention output — `weights @ V`
- [ ] RoPE in Python as a reference
- [ ] Build setup (`setup.py` / `torch.utils.cpp_extension`)
- [ ] CUDA attention kernel
- [ ] RoPE applied inside the kernel
- [ ] KV-cache append + incremental decode
- [ ] Correctness tests vs. PyTorch
- [ ] Benchmarks across context lengths
