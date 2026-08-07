"""
Implement a RoPE kernel
"""

import os
os.environ["TRITON_CACHE_DIR"] = "/jumbo/lisp/f007twf/.triton_cache"

import math

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
LOG_10000 = math.log(10000)

@triton.jit
def rope_kernel(x_ptr,   # (batch, n_heads, seq_len, head_dim)
                out_ptr,
                x_row_stride,
                head_stride,
                batch_stride,
                out_row_stride,
                out_head_stride,
                out_batch_stride,
                log_10000,
                half: tl.constexpr,     # head_dim // 2 (compile-time constant)
                BLOCK_SIZE: tl.constexpr):

    # One program per (batch, head, token position). This program computes the
    # cos/sin for its position and applies the 2D rotation to each head_dim pair.
    batch = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.program_id(2)

    h = tl.arange(0, BLOCK_SIZE // 2)
    mask = h < half

    # Interleaved pairing: element 2i and 2i+1 form one rotated pair.
    even_offsets = 2 * h
    odd_offsets = 2 * h + 1

    even_x_ptrs = x_ptr + (batch * batch_stride) + (head * head_stride) + (x_row_stride * seq) + even_offsets
    odd_x_ptrs = x_ptr + (batch * batch_stride) + (head * head_stride) + (x_row_stride * seq) + odd_offsets

    even_x = tl.load(even_x_ptrs, mask=mask)
    odd_x = tl.load(odd_x_ptrs, mask=mask)

    # Compute cos/sin for this position (seq) across the head_dim pairs.
    # theta_i = seq * base^(-2i/d) = seq * exp( -(i/half) * ln(base) )
    scale = -log_10000 / half
    inv_freq = tl.exp(h.to(tl.float32) * scale)
    theta = seq.to(tl.float32) * inv_freq
    cos_t = tl.cos(theta)
    sin_t = tl.sin(theta)

    even_out = even_x * cos_t - odd_x * sin_t
    odd_out = even_x * sin_t + odd_x * cos_t

    # Cast back to the output dtype (loads promote fp16*fp32 -> fp32).
    even_out = even_out.to(out_ptr.dtype.element_ty)
    odd_out = odd_out.to(out_ptr.dtype.element_ty)

    even_out_ptrs = out_ptr + (batch * out_batch_stride) + (head * out_head_stride) + (seq * out_row_stride) + even_offsets
    odd_out_ptrs = out_ptr + (batch * out_batch_stride) + (head * out_head_stride) + (seq * out_row_stride) + odd_offsets

    tl.store(even_out_ptrs, even_out, mask=mask)
    tl.store(odd_out_ptrs, odd_out, mask=mask)


def rope(x):
    assert x.ndim == 4, "expected a (batch, n_heads, seq_len, head_dim) input"
    assert x.is_contiguous(), "kernel assumes row-major contiguous input"

    batch, n_heads, seq_len, dim = x.shape
    assert dim % 2 == 0, "head_dim must be even for paired rotation"

    half = dim // 2

    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(dim)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    # One program per (batch, head, position); the kernel handles all the pairs
    # of a head_dim row itself. Last dim (head_dim) is assumed contiguous, so its
    # stride is 1 and doesn't need to be passed.
    grid = (batch, n_heads, seq_len)
    rope_kernel[grid](
        x, out,
        x.stride(2), x.stride(1), x.stride(0),           # seq / head / batch strides
        out.stride(2), out.stride(1), out.stride(0),
        LOG_10000,
        half=half,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return out


def _apply_rope_torch(x, base):
    # Shared torch implementation of the interleaved rotation over the last dim
    # of a (batch, n_heads, seq_len, head_dim) tensor. cos/sin are (seq_len, half)
    # and broadcast over the batch/head axes.
    batch, n_heads, seq_len, dim = x.shape
    half = dim // 2

    x_f = x.float()
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32).unsqueeze(1)
    indices = torch.arange(half, device=x.device, dtype=torch.float32)
    theta = positions * (base ** (-indices / half))  # (seq_len, half)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    x_even = x_f[..., 0::2]  # (batch, n_heads, seq_len, half)
    x_odd = x_f[..., 1::2]

    out_even = x_even * cos_t - x_odd * sin_t
    out_odd = x_even * sin_t + x_odd * cos_t
    return out_even, out_odd


def rope_reference(x, base=10000.0):
    # Independent of the kernel path (recomputes cos/sin from scratch).
    out_even, out_odd = _apply_rope_torch(x, base)
    out = torch.empty_like(x, dtype=torch.float32)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out.to(x.dtype)


def rope_naive(x, base=10000.0):
    """Idiomatic plain-PyTorch RoPE apply. Computes cos/sin the same way the
    kernel does in-kernel, so benchmark_vs_naive compares equal work."""
    out_even, out_odd = _apply_rope_torch(x, base)
    # stack + flatten re-interleaves: result[..., 2i] = even_i, [..., 2i+1] = odd_i
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(x.dtype)


# (batch, n_heads, seq_len, head_dim). Shapes stress masking: non-power-of-2
# head_dim (100), odd seq_len (257), small batch/head counts.
DEFAULT_SHAPES = (
    (2, 4, 128, 64),
    (1, 8, 257, 100),
    (4, 8, 512, 128),
    (1, 2, 16, 256),
)


def check_correctness(shapes=DEFAULT_SHAPES, dtype=torch.float16):
    print(f"{'shape (B,H,S,D)':>20} | {'max_err':>10} | {'status':>6}")

    all_ok = True
    for shape in shapes:
        x = torch.randn(shape, device=DEVICE, dtype=dtype)

        out = rope(x)
        ref = rope_reference(x)

        max_err = (out.float() - ref.float()).abs().max().item()
        ok = max_err < 2e-2  # fp16 has ~1e-3 relative precision; leave headroom
        all_ok &= ok

        print(f"{str(shape):>20} | {max_err:>10.5f} | {'OK' if ok else 'FAIL':>6}")

    return all_ok


# Set to your GPU's datasheet peak (GB/s) to see a %-of-peak column; None hides it.
PEAK_BANDWIDTH_GBPS = None

# Realistic transformer shapes: fixed batch/heads/head_dim, sweeping seq_len.
BENCH_SHAPES = (
    (4, 32, 512, 128),
    (4, 32, 1024, 128),
    (4, 32, 2048, 128),
    (4, 32, 4096, 128),
    (2, 32, 8192, 128),
)


def benchmark_bandwidth(shapes=BENCH_SHAPES, dtype=torch.float16):
    x_bytes = torch.tensor([], dtype=dtype).element_size()

    header = f"{'shape (B,H,S,D)':>20} | {'time (ms)':>10} | {'GB/s':>10}"
    if PEAK_BANDWIDTH_GBPS:
        header += f" | {'% peak':>8}"
    print(header)

    results = []
    for shape in shapes:
        x = torch.randn(shape, device=DEVICE, dtype=dtype)

        ms = triton.testing.do_bench(lambda: rope(x))

        # cos/sin are computed in-kernel now, so traffic is just x read + out write.
        bytes_moved = 2 * x.numel() * x_bytes
        gbps = bytes_moved / (ms * 1e-3) / 1e9

        row = f"{str(shape):>20} | {ms:>10.4f} | {gbps:>10.2f}"
        if PEAK_BANDWIDTH_GBPS:
            row += f" | {gbps / PEAK_BANDWIDTH_GBPS * 100:>7.1f}%"
        print(row)

        results.append((shape, ms, gbps))

    return results


def benchmark_vs_naive(shapes=BENCH_SHAPES, dtype=torch.float16):
    header = (f"{'shape (B,H,S,D)':>20} | {'triton (ms)':>12} | "
              f"{'naive (ms)':>12} | {'speedup':>8} | {'max_err':>8}")
    print(header)
    print("-" * len(header))

    results = []
    for shape in shapes:
        x = torch.randn(shape, device=DEVICE, dtype=dtype)

        triton_ms = triton.testing.do_bench(lambda: rope(x))
        naive_ms = triton.testing.do_bench(lambda: rope_naive(x))
        speedup = naive_ms / triton_ms

        # Sanity: the two paths should agree (both apply the same rotation).
        max_err = (rope(x).float() - rope_naive(x).float()).abs().max().item()

        print(f"{str(shape):>20} | {triton_ms:>12.4f} | "
              f"{naive_ms:>12.4f} | {speedup:>7.2f}x | {max_err:>8.4f}")

        results.append((shape, triton_ms, naive_ms, speedup))

    return results


if __name__ == "__main__":
    check_correctness()
    print()
    benchmark_bandwidth()
    print()
    benchmark_vs_naive()