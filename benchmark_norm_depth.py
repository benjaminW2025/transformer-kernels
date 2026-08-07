"""
How much does fusion save us across the normalization ops of a 1.5B param model
"""

import os
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO, "kernels"))

import torch
import triton

from kernels.fused_rms import (  # noqa: E402  (path setup must precede the import)
    DEVICE,
    rms_norm,
    rms_norm_naive,
    rms_norm_compiled,
)

# Qwen2.5-1.5B geometry: 28 decoder layers, hidden size 1536.
N_LAYERS = 28
N_NORMS = 2 * N_LAYERS + 1
HIDDEN = 1536

# Rows = tokens in the batch (batch * seq_len flattened, as RMSNorm sees them).
TOKEN_COUNTS = (2048, 4096, 8192)

# L40S datasheet peak. Set to None to hide the %-of-peak column.
PEAK_BANDWIDTH_GBPS = 864.0

EPSILON = 1e-6


def _run_hot(fn, x, weight, n_norms=N_NORMS):
    # Each output feeds the next norm, so the allocator ping-pongs between two
    # buffers and the whole chain stays within one activation's footprint.
    for _ in range(n_norms):
        x = fn(x, weight, EPSILON)
    return x


def _run_cold(fn, xs, weight):
    # Returning the list keeps all 57 outputs alive for the duration of the
    # call. Without that the freed output block gets handed straight back on
    # the next iteration and the writes stay resident in L2.
    return [fn(x, weight, EPSILON) for x in xs]


def _bench(fn, *args):
    fn(*args)  # force Triton / Inductor compilation outside the timed region
    torch.cuda.synchronize()
    return triton.testing.do_bench(lambda: fn(*args))


def _print_table(title, rows, dtype_bytes):
    print(title)
    header = (f"{'tokens':>8} | {'triton (ms)':>12} | {'compiled (ms)':>14} | "
              f"{'naive (ms)':>12} | {'vs naive':>9} | {'vs compiled':>12} | "
              f"{'triton GB/s':>12}")
    if PEAK_BANDWIDTH_GBPS:
        header += f" | {'% peak':>7}"
    print(header)
    print("-" * len(header))

    for tokens, triton_ms, compiled_ms, naive_ms in rows:
        # 57 norms, each reading x and writing out once at the input dtype.
        bytes_moved = N_NORMS * 2 * tokens * HIDDEN * dtype_bytes
        gbps = bytes_moved / (triton_ms * 1e-3) / 1e9

        line = (f"{tokens:>8} | {triton_ms:>12.4f} | {compiled_ms:>14.4f} | "
                f"{naive_ms:>12.4f} | {naive_ms / triton_ms:>8.2f}x | "
                f"{compiled_ms / triton_ms:>11.2f}x | {gbps:>12.2f}")
        if PEAK_BANDWIDTH_GBPS:
            line += f" | {gbps / PEAK_BANDWIDTH_GBPS * 100:>6.1f}%"
        print(line)
    print()


def benchmark(token_counts=TOKEN_COUNTS, dtype=torch.float16):
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()

    hot_rows = []
    cold_rows = []

    for tokens in token_counts:
        # Ones is the standard RMSNorm init, and it makes the norm idempotent
        # so the 57-deep hot chain cannot drift or overflow in fp16. Timing is
        # value-independent here (no data-dependent branches), so this costs
        # nothing in realism.
        weight = torch.ones((HIDDEN,), device=DEVICE, dtype=dtype)

        x = torch.randn((tokens, HIDDEN), device=DEVICE, dtype=dtype)
        hot_rows.append((
            tokens,
            _bench(_run_hot, rms_norm, x, weight),
            _bench(_run_hot, rms_norm_compiled, x, weight),
            _bench(_run_hot, rms_norm_naive, x, weight),
        ))
        del x

        xs = [torch.randn((tokens, HIDDEN), device=DEVICE, dtype=dtype)
              for _ in range(N_NORMS)]
        cold_rows.append((
            tokens,
            _bench(_run_cold, rms_norm, xs, weight),
            _bench(_run_cold, rms_norm_compiled, xs, weight),
            _bench(_run_cold, rms_norm_naive, xs, weight),
        ))
        del xs, weight
        torch.cuda.empty_cache()

    print(f"RMSNorm x{N_NORMS} (28-layer 1.5B), hidden={HIDDEN}, {dtype}\n")
    _print_table("HOT  - same tensor rechained, working set fits in cache",
                 hot_rows, dtype_bytes)
    _print_table("COLD - 57 distinct tensors, every access misses cache",
                 cold_rows, dtype_bytes)

    print("Bound on the fusion win:")
    for (tokens, h_tri, _, h_naive), (_, c_tri, _, c_naive) in zip(hot_rows, cold_rows):
        print(f"  {tokens:>6} tokens: {h_naive / h_tri:>5.2f}x hot "
              f"-> {c_naive / c_tri:>5.2f}x cold")
    print("\nThe hot number is what fusion buys when activations stay cached;")
    print("the cold number is the realistic per-forward-pass ceiling, before")
    print("attenuating by the norms' share of total model runtime.")

    return hot_rows, cold_rows


def check_correctness(tokens=4096, dtype=torch.float16):
    """Confirm the three implementations still agree after a 57-deep chain."""
    weight = torch.ones((HIDDEN,), device=DEVICE, dtype=dtype)
    x = torch.randn((tokens, HIDDEN), device=DEVICE, dtype=dtype)

    tri = _run_hot(rms_norm, x, weight).float()
    nai = _run_hot(rms_norm_naive, x, weight).float()
    com = _run_hot(rms_norm_compiled, x, weight).float()

    err_naive = (tri - nai).abs().max().item()
    err_compiled = (tri - com).abs().max().item()

    print(f"after {N_NORMS} chained norms ({tokens} tokens):")
    print(f"  max |triton - naive|    = {err_naive:.5f}")
    print(f"  max |triton - compiled| = {err_compiled:.5f}")

    ok = err_naive < 2e-2 and err_compiled < 2e-2
    print(f"  status: {'OK' if ok else 'FAIL'}\n")
    return ok


if __name__ == "__main__":
    check_correctness()
    benchmark()
