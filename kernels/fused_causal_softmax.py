"""
Causal fused softmax
"""

import os
os.environ["TRITON_CACHE_DIR"] = "/jumbo/lisp/f007twf/.triton_cache"

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def online_softmax(x_ptr,   # (batch, n_heads, seq_len, seq_len) attention scores
                   out_ptr,  # (batch, n_heads, seq_len, seq_len)
                   x_row_stride,
                   x_head_stride,
                   x_batch_stride,
                   out_row_stride,
                   out_head_stride,
                   out_batch_stride,
                   seq_len,
                   BLOCK_SIZE: tl.constexpr
                   ):
    # logic will handle for (seq_len dim)
    # apply stride to manage batch dim

    # Indexing
    batch = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.program_id(2) # Token index within the sequence

    # Compute offsets and pointers (row is selected by the token index `seq`)
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + (batch * x_batch_stride) + (head * x_head_stride) + (seq * x_row_stride) + offsets

    # Causal mask: query position `seq` attends only to keys 0..seq (inclusive).
    causal_mask = offsets < seq + 1
    # Bounds mask: which offsets are real memory (BLOCK_SIZE may exceed seq_len).
    bounds_mask = offsets < seq_len

    x = tl.load(x_ptrs, mask=causal_mask, other=float("-inf"))

    # Subtract max
    row_max = tl.max(x, axis=0)
    exp = tl.exp(x - row_max)
    output = exp / tl.sum(exp, axis=0)   # already 0 at causal-masked positions

    # Store
    out_ptrs = out_ptr + (batch * out_batch_stride) + (head * out_head_stride) + (seq * out_row_stride) + offsets
    tl.store(out_ptrs, output, mask=bounds_mask)


def causal_softmax(x):
    assert x.ndim == 4, "expected (batch, n_heads, seq_len, seq_len) attention scores"
    assert x.is_contiguous(), "kernel assumes row-major contiguous input"

    batch, n_heads, q_len, k_len = x.shape
    assert q_len == k_len, "causal softmax expects a square (seq_len x seq_len) score matrix"
    seq_len = q_len

    out = torch.empty_like(x)

    # Assumes row can fit into single block
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    grid = (batch, n_heads, seq_len)
    online_softmax[grid](
        x, out,
        x.stride(2), x.stride(1), x.stride(0),   # row (query) / head / batch strides
        out.stride(2), out.stride(1), out.stride(0),
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return out


def _causal_mask(seq_len, device):
    # True above the diagonal: the positions a causal query may NOT attend to.
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


def causal_softmax_reference(x):
    # Independent fp32 oracle.
    seq_len = x.shape[-1]
    scores = x.float().masked_fill(_causal_mask(seq_len, x.device), float("-inf"))
    return torch.softmax(scores, dim=-1).to(x.dtype)


def causal_softmax_naive(x):
    # Idiomatic plain-PyTorch causal softmax in the input dtype - a fair throughput
    # baseline for the speedup comparison (masks + softmaxes on every call, as the
    # kernel also does its causal masking every call).
    seq_len = x.shape[-1]
    return torch.softmax(x.masked_fill(_causal_mask(seq_len, x.device), float("-inf")), dim=-1)


# (batch, n_heads, seq_len, seq_len). Shapes stress masking: non-power-of-2
# seq_len (100), odd seq_len (65), small counts.
DEFAULT_SHAPES = (
    (2, 4, 64, 64),
    (1, 8, 100, 100),
    (2, 4, 128, 128),
    (1, 2, 65, 65),
)


def check_correctness(shapes=DEFAULT_SHAPES, dtype=torch.float16):
    print(f"{'shape (B,H,S,S)':>22} | {'max_err':>10} | {'status':>6}")

    all_ok = True
    for shape in shapes:
        x = torch.randn(shape, device=DEVICE, dtype=dtype)

        out = causal_softmax(x)
        ref = causal_softmax_reference(x)

        max_err = (out.float() - ref.float()).abs().max().item()
        ok = max_err < 2e-3  # softmax outputs live in [0, 1]; a tight tolerance is fine
        all_ok &= ok

        print(f"{str(shape):>22} | {max_err:>10.6f} | {'OK' if ok else 'FAIL':>6}")

    return all_ok


# Set to your GPU's datasheet peak (GB/s) to see a %-of-peak column; None hides it.
PEAK_BANDWIDTH_GBPS = None


def _effective_bytes(shape, dtype_bytes):
    # Causal load reads the lower triangle (row i reads i+1 keys); the store writes
    # the full row (probs plus the above-diagonal zeros). So reads ~= half the
    # matrix, writes = the full matrix - not the naive 2*numel.
    batch, n_heads, seq_len, _ = shape
    read_elems = batch * n_heads * seq_len * (seq_len + 1) / 2
    write_elems = batch * n_heads * seq_len * seq_len
    return (read_elems + write_elems) * dtype_bytes


def _bench_one(shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    triton_ms = triton.testing.do_bench(lambda: causal_softmax(x))
    naive_ms = triton.testing.do_bench(lambda: causal_softmax_naive(x))
    speedup = naive_ms / triton_ms

    gbps = _effective_bytes(shape, x.element_size()) / (triton_ms * 1e-3) / 1e9
    max_err = (causal_softmax(x).float() - causal_softmax_reference(x).float()).abs().max().item()

    return triton_ms, gbps, naive_ms, speedup, max_err


def _print_sweep(title, label, shapes, dtype):
    print(title)
    cols = f"{label:>8} | {'triton (ms)':>12} | {'GB/s':>10}"
    if PEAK_BANDWIDTH_GBPS:
        cols += f" | {'% peak':>7}"
    cols += f" | {'naive (ms)':>12} | {'speedup':>8} | {'max_err':>9}"
    print(cols)
    print("-" * len(cols))

    for shape, key in shapes:
        triton_ms, gbps, naive_ms, speedup, max_err = _bench_one(shape, dtype)
        row = f"{key:>8} | {triton_ms:>12.4f} | {gbps:>10.2f}"
        if PEAK_BANDWIDTH_GBPS:
            row += f" | {gbps / PEAK_BANDWIDTH_GBPS * 100:>6.1f}%"
        row += f" | {naive_ms:>12.4f} | {speedup:>7.2f}x | {max_err:>9.6f}"
        print(row)
    print()


def benchmark(batch=4, dtype=torch.float16):
    # Sweep seq_len at fixed n_heads.
    n_heads_fixed = 16
    seq_shapes = [((batch, n_heads_fixed, s, s), s) for s in (128, 256, 512, 1024, 2048)]
    _print_sweep(f"Sweep seq_len (batch={batch}, n_heads={n_heads_fixed}):",
                 "seq_len", seq_shapes, dtype)

    # Sweep n_heads at fixed seq_len.
    seq_len_fixed = 1024
    head_shapes = [((batch, h, seq_len_fixed, seq_len_fixed), h) for h in (4, 8, 16, 32, 64)]
    _print_sweep(f"Sweep n_heads (batch={batch}, seq_len={seq_len_fixed}):",
                 "n_heads", head_shapes, dtype)


if __name__ == "__main__":
    check_correctness()
    print()
    benchmark()

