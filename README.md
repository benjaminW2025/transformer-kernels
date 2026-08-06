# Transformer Kernels

(Comprehensive write up is TBD, results still need to be cleaned up + written to this README)

This repo contains my implementation of kernels for several key transformer components in Triton, along with some learning notes from the past three weeks. All kernels were hand implemented to build fluency in memory indexing, memory guards, and the Triton stack. Tests are benchmarked on the NVIDIA L40 Lovelace architecture with more details provided below. While kernels are all hand implemented, the benchmarking was written with AI assistance.

## Set up

Download necessary environment packages and run each script independently. Make sure to edit the manual Triton cache file path prepend which I added to each kernel since my home directory on the lab machines I was using were full.

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Kernels

We implement kernels for the following operations:

```
kernels/
├── tiled_matmul.py          # Tiled matrix multiplication (autotuned, GROUP_M swizzle)
├── fused_rms.py             # Fused RMSNorm
├── rope.py                  # RoPE (rotary position embeddings)
├── fused_causal_softmax.py  # Fused causal softmax
└── flash_attention.py       # FlashAttention forward pass (TBD)
```

Each kernel file is self-contained: the Triton kernel, a host launcher, a
PyTorch reference for correctness, and a benchmark harness (bandwidth / TFLOP/s
and, where relevant, a speedup comparison against a naive baseline).

## Results

**Fused RMSNorm:**
RMSNorm computes $\mathrm{RMSNorm}(x) = \frac{x}{\mathrm{RMS}(x)} \cdot \gamma$ where $\mathrm{RMS}(x) = \sqrt{\frac{1}{N} \cdot \sum x_i^2}$ and $\gamma$ is a learned weight. For $x\in \mathbb{R}^{M\times N}$, the naive implementation performs six operations (`.float()`, .`pow()`, `.mean()`, two element wise multiplications, `.to()`). Each of `.pow` and the multiplications then perform a total of $24MN$ bytes of HBM access at fp32. `.float()` and `.to()` contribute $12MN$ (read fp16, write fp32, and vice versa), and `.mean()` contributes $4MN+4M$ (it only writes $M$ means, one per row). In total, the naive implementation performs $40MN + M$ DRAM accesses. Our fused kernel reads $2MN$ bytes and writes $2MN$ bytes, for a theoretical speed up of ~10x. We benchmark against both the naive implementation, and PyTorch fused kernel using torch.compile, and empirically see the speed up against the naive RMSNorm and competitive performance against the fused PyTorch kernel:

|     N | Triton (ms) | Compiled (ms) | Naive (ms) | vs Compiled | vs Naive | Max Error |
| ----: | ----------: | ------------: | ---------: | ----------: | -------: | --------: |
|   512 |      0.0279 |        0.0311 |     0.1074 |       1.11× |    3.84× |    0.0039 |
|  1024 |      0.0544 |        0.0307 |     0.1683 |       0.56× |    3.09× |    0.0039 |
|  2048 |      0.1068 |        0.1086 |     0.7162 |       1.02× |    6.70× |    0.0039 |
|  4096 |      0.1900 |        0.2084 |     1.9675 |       1.10× |   10.36× |    0.0039 |
|  8192 |      0.4150 |        0.4077 |     4.0614 |       0.98× |    9.79× |    0.0039 |
| 16384 |      0.8232 |        0.8229 |     8.1401 |       1.00× |    9.89× |    0.0078 |

**Fused Causal Softmax:**

Fusing the softmax operation leads to an approximately $4\times$ speedup over the naive implementation. For an input $x\in\mathbb{R}^{MN}$, the naive implementation 1) Computes the max element by reading the $MN$ elements and writing $M$ elements 2) Subtracts the max element from each row by reading $MN + M$ elements and writing $MN$ elements 3) Exponentiating each element by reading $MN$ elements and writing $MN$ elements 4) Computing the denominator by reading $MN$ elements and writing $M$ elements 5) Dividing by reading $MN + M$ elements and writing $MN$ elements. This totals to $8MN + 4M$ elements. Our fused kernel reads in the $MN$ elements once, and writes them out once, for an $\sim 4\times$ speedup which is reflected in our benchmarking.

Sweep `seq_len` (batch = 4, n_heads = 16)
| Seq Length | Triton (ms) |   GB/s | Naive (ms) | Speedup | Max Error |
| ---------: | ----------: | -----: | ---------: | ------: | --------: |
|        128 |      0.0101 | 310.95 |     0.0272 |   2.68× |  0.000061 |
|        256 |      0.0285 | 442.38 |     0.0565 |   1.99× |  0.000122 |
|        512 |      0.0986 | 510.73 |     0.1744 |   1.77× |  0.000061 |
|       1024 |      0.3467 | 580.84 |     1.2390 |   3.57× |  0.000031 |
|       2048 |      1.2663 | 636.07 |     4.9319 |   3.89× |  0.000031 |

Sweep `n_heads` (batch = 4, seq_len = 1024)
| Number of Heads | Triton (ms) |   GB/s | Naive (ms) | Speedup | Max Error |
| --------------: | ----------: | -----: | ---------: | ------: | --------: |
|               4 |      0.1032 | 487.80 |     0.1790 |   1.73× |  0.000061 |
|               8 |      0.1806 | 557.54 |     0.4346 |   2.41× |  0.000031 |
|              16 |      0.3477 | 579.18 |     1.2372 |   3.56× |  0.000061 |
|              32 |      0.6696 | 601.51 |     2.4506 |   3.66× |  0.000061 |
|              64 |      1.3426 | 600.02 |     4.8928 |   3.64× |  0.000122 |

**RoPE:**

Need to clean results

**Tiled Matrix Multiplication:**

Need to clean results



### Minor notes

- On tiled matrix multiplication, cuBLAS implementation ops for an sub optimal tiling config leading to poor performance and more HBM accesses compared to my autotuned config; our kernel achieves ~80% of the theoretical max
- The L40's L2 cache is large enough to fit the entire $4096^2$ matrix at fp16 precision, which is one reason I suspect the empirical FLOPS/byte is ~3 times higher than the theoretical maximum computed with using just DRAM access speed (i.e., the matrix gets loaded into L2 cache and saves time on what otherwise would be HBM access)
- I wrote the RoPE kernel to handle inputs with tensor dimensions of (batch, num_heads, seq_len, d_heads) to be consistent with how RoPE is applied within attention computation
- The base softmax kernel follows the Triton documentation but ```fused_causal_softmax.py``` natively handles (batch, num_heads, seq_len, d_heads) for attention computation and causal masking

### TODO
- [X] Standard transformer kernels
- [ ] Write up prelim benchmarking results to README
- [ ] Generalize matrix multiplication beyond 2D tensors
- [ ] Optimize fused causal softmax for longer sequences
- [ ] Online softmax + FlashAttention forward pass
- [ ] Online softmax recurrence proof for write up
- [ ] Profile naive vs fused operations overhead
- [ ] Comprehensive write up