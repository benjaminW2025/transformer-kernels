"""
Implement a tiled matrix multiplication kernel and analyze peak memory bandwidth
"""

import os
os.environ["TRITON_CACHE_DIR"] = "/jumbo/lisp/f007twf/.triton_cache"

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def tiled_matmul(x_ptr, # x is a (A, C) matrix
                 y_ptr, # y is a (C, B) matrix
                 output_ptr,
                 A,
                 B,
                 C,
                 x_row_stride,
                 y_row_stride,
                 output_row_stride,
                 BLOCK_SIZE: tl.constexpr,
                 num_stages: tl.constexpr):
    # 2D launch grid
    # Get PID along axes to get blockid
    block_x = tl.program_id(axis=0)
    block_y = tl.program_id(axis=1)

    # Accumulator array - fp32 for numerical stability in tl.dot, cast down on store
    acc = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)

    x_row_coords = block_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    y_col_offsets = block_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    k_offsets = tl.arange(0, BLOCK_SIZE)

    # Iteratively load a BLOCK_SIZE by BLOCK_SIZE block (zero padded at the edges)
    # and accumulate the partial dot product along the contraction dim C
    for i in tl.range(0, tl.cdiv(C, BLOCK_SIZE), num_stages=num_stages):
        x_col_offsets = i * BLOCK_SIZE + k_offsets
        y_row_coords = i * BLOCK_SIZE + k_offsets

        # Need to broadcast downwards
        x_offsets = (x_row_coords[:, None] * x_row_stride) + x_col_offsets[None, :]
        y_offsets = (y_row_coords[:, None] * y_row_stride) + y_col_offsets[None, :]

        # Compute mask
        x_mask = (x_row_coords[:, None] < A) & (x_col_offsets[None, :] < C)
        y_mask = (y_row_coords[:, None] < C) & (y_col_offsets[None, :] < B)

        x_ptrs = x_ptr + x_offsets
        y_ptrs = y_ptr + y_offsets

        # Load in x and y
        x_shared = tl.load(x_ptrs, mask=x_mask, other=0.0)
        y_shared = tl.load(y_ptrs, mask=y_mask, other=0.0)

        # Accumulator and dot product
        acc = tl.dot(x_shared, y_shared, acc)

    # Write back out
    output_row_coords = block_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_col_coords = block_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    output_ptrs = output_ptr + (output_row_coords[:, None] * output_row_stride) + output_col_coords[None, :]
    out_mask = (output_row_coords[:, None] < A) & (output_col_coords[None, :] < B)

    # Store output (pointers, data tensor, mask)
    tl.store(output_ptrs, acc.to(tl.float16), mask=out_mask)

properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REG = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()


def matmul(x, y, BLOCK_SIZE=128):
    assert x.ndim == 2 and y.ndim == 2 and x.shape[1] == y.shape[0], "incompatible dimensions"
    assert x.is_contiguous() and y.is_contiguous(), "matmul kernel assumes row-major contiguous inputs"

    A, C = x.shape
    _, B = y.shape

    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Create output
    output = torch.empty((A, B), device=x.device, dtype=torch.float16)

    # 2D launch grid: one program per (BLOCK_SIZE x BLOCK_SIZE) output tile
    grid = (triton.cdiv(A, BLOCK_SIZE), triton.cdiv(B, BLOCK_SIZE))

    tiled_matmul[grid](
        x, y, output,
        A, B, C,
        x.stride(0), y.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
    )

    return output


def benchmark_block_sizes(A=4096, B=4096, C=4096, block_sizes=(16, 32, 64, 128, 256, 512)):
    x = torch.randn((A, C), device=DEVICE, dtype=torch.float16)
    y = torch.randn((C, B), device=DEVICE, dtype=torch.float16)

    ref = torch.matmul(x, y)

    print(f"{'BLOCK_SIZE':>10} | {'time (ms)':>10} | {'TFLOP/s':>10} | {'max_err':>10}")
    results = []
    for block_size in block_sizes:
        try:
            out = matmul(x, y, BLOCK_SIZE=block_size)
        except triton.runtime.errors.OutOfResources as e:
            print(f"{block_size:>10} | skipped (out of resources: {e})")
            continue

        max_err = (out.float() - ref.float()).abs().max().item()

        ms = triton.testing.do_bench(lambda: matmul(x, y, BLOCK_SIZE=block_size))
        tflops = 2 * A * B * C / (ms * 1e-3) / 1e12

        results.append((block_size, ms, tflops, max_err))
        print(f"{block_size:>10} | {ms:>10.4f} | {tflops:>10.2f} | {max_err:>10.4f}")

    return results


def estimate_smem_bytes(block_m, block_n, block_k, num_stages, dtype_bytes=2):
    # Pipelined loads keep `num_stages` in-flight tiles of x (BLOCK_M x BLOCK_K)
    # and y (BLOCK_K x BLOCK_N) resident in shared memory at once.
    return num_stages * (block_m * block_k + block_k * block_n) * dtype_bytes


# A small, hand-picked set instead of a full cartesian product. Each config is
# compiled AND timed during autotuning, so keeping this short keeps tuning fast.
# Rationale for these choices on this GPU (Turing / RTX 8000, 64 KB SMEM):
#   - BLOCK_K stays small (32/64): it drives SMEM and the fp32 accumulator is
#     BLOCK_M x BLOCK_N regardless, so a fat K just wastes shared memory.
#   - Output tile (BLOCK_M x BLOCK_N) kept <= 128x128: a 128x128 fp32 acc is
#     already 64 KB of registers and spills hard on Turing.
#   - num_stages = 2: Turing has no cp.async, so deeper pipelining mostly costs
#     SMEM without hiding latency.
#   - num_warps scaled with tile area (more work per block -> more warps).
# Tuned for L40 (Ada Lovelace, sm_89): ~100 KB SMEM/SM and it HAS cp.async, so
# unlike Turing, deep pipelining (num_stages 3-4) hides global-load latency and
# large tiles are worth it. Guidance baked into the list:
#   - Large output tiles (up to 128x256 / 256x128) for high arithmetic intensity.
#   - BLOCK_K 32/64; num_stages raised to 3-4 where SMEM allows (cp.async wins).
#   - num_warps scaled with tile area (8 for 128x128+, 4 for mid, 2 for small).
#   - SMEM per config stays <= ~96 KB so it fits the 100 KB opt-in budget; the
#     estimate_smem_bytes guard prunes anything too big for the actual device.
_CANDIDATE_CONFIGS = [
    # (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
    (128, 256, 64, 2, 8),   # 96 KB
    (256, 128, 64, 2, 8),   # 96 KB
    (128, 128, 64, 3, 8),   # 96 KB
    (128, 128, 32, 4, 4),   # 64 KB
    (128, 64,  64, 4, 4),   # 96 KB
    (64,  128, 64, 4, 4),   # 96 KB
    (128, 64,  32, 4, 4),   # 48 KB
    (64,  64,  32, 4, 2),   # 32 KB - small-shape fallback
]


# Default group width for the autotuned kernel. GROUP_M reorders the 1D program
# id into column-major-ish groups of GROUP_M block-rows, so the blocks running
# concurrently reuse the same slices of x and y out of L2 instead of streaming a
# whole row of y per block. GROUP_M=1 is plain row-major ordering (no swizzle).
DEFAULT_GROUP_M = 8


def get_autotune_configs(group_m=DEFAULT_GROUP_M):
    configs = []
    for block_m, block_n, block_k, num_stages, num_warps in _CANDIDATE_CONFIGS:
        # Guard so the list stays valid if it's ever run on a smaller-SMEM GPU.
        if estimate_smem_bytes(block_m, block_n, block_k, num_stages) > SIZE_SMEM:
            continue
        configs.append(
            triton.Config(
                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k,
                 "GROUP_M": group_m},
                num_stages=num_stages,
                num_warps=num_warps,
            )
        )
    return configs


@triton.autotune(configs=get_autotune_configs(), key=["A", "B", "C"])
@triton.jit
def tiled_matmul_mnk(x_ptr, # x is a (A, C) matrix
                      y_ptr, # y is a (C, B) matrix
                      output_ptr,
                      A,
                      B,
                      C,
                      x_row_stride,
                      y_row_stride,
                      output_row_stride,
                      BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr,
                      GROUP_M: tl.constexpr):
    # 1D launch grid, remapped to a (block_m, block_n) tile in grouped order.
    # Programs are walked GROUP_M rows at a time down a column of the output, so
    # a wave of concurrently-resident blocks touches GROUP_M row-panels of x and
    # ~(wave / GROUP_M) column-panels of y - both small enough to stay hot in L2.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(A, BLOCK_M)
    num_pid_n = tl.cdiv(B, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    # Last group may be short if num_pid_m isn't a multiple of GROUP_M.
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group

    block_m = first_pid_m + (pid_in_group % group_size_m)
    block_n = pid_in_group // group_size_m

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    x_row_coords = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    y_col_offsets = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    # `num_stages`/`num_warps` aren't kernel args here - triton.autotune passes
    # them as compiler hints that software-pipeline this loop automatically.
    for i in range(0, tl.cdiv(C, BLOCK_K)):
        x_col_offsets = i * BLOCK_K + k_offsets
        y_row_coords = i * BLOCK_K + k_offsets

        x_offsets = (x_row_coords[:, None] * x_row_stride) + x_col_offsets[None, :]
        y_offsets = (y_row_coords[:, None] * y_row_stride) + y_col_offsets[None, :]

        x_mask = (x_row_coords[:, None] < A) & (x_col_offsets[None, :] < C)
        y_mask = (y_row_coords[:, None] < C) & (y_col_offsets[None, :] < B)

        x_ptrs = x_ptr + x_offsets
        y_ptrs = y_ptr + y_offsets

        x_shared = tl.load(x_ptrs, mask=x_mask, other=0.0)
        y_shared = tl.load(y_ptrs, mask=y_mask, other=0.0)

        acc = tl.dot(x_shared, y_shared, acc)

    output_row_coords = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_col_coords = block_n * BLOCK_N + tl.arange(0, BLOCK_N)

    output_ptrs = output_ptr + (output_row_coords[:, None] * output_row_stride) + output_col_coords[None, :]
    out_mask = (output_row_coords[:, None] < A) & (output_col_coords[None, :] < B)

    tl.store(output_ptrs, acc.to(tl.float16), mask=out_mask)


def matmul_mnk(x, y, config=None):
    """
    Matrix multiplication
    """
    assert x.ndim == 2 and y.ndim == 2 and x.shape[1] == y.shape[0], "incompatible dimensions"
    assert x.is_contiguous() and y.is_contiguous(), "matmul kernel assumes row-major contiguous inputs"

    A, C = x.shape
    _, B = y.shape

    output = torch.empty((A, B), device=x.device, dtype=torch.float16)

    # BLOCK_M/BLOCK_N are chosen by the autotuner at call time, so the grid
    # has to be a function of the winning config's meta-parameters. 1D grid:
    # the kernel does the (m, n) remap itself so it can group by GROUP_M.
    grid = lambda META: (triton.cdiv(A, META["BLOCK_M"]) * triton.cdiv(B, META["BLOCK_N"]),)

    if config is None:
        tiled_matmul_mnk[grid](
            x, y, output,
            A, B, C,
            x.stride(0), y.stride(0), output.stride(0),
        )
    else:
        # `.fn` is the undecorated JITFunction, so this bypasses autotuning and
        # runs exactly the config we hand it.
        tiled_matmul_mnk.fn[grid(config.kwargs)](
            x, y, output,
            A, B, C,
            x.stride(0), y.stride(0), output.stride(0),
            **config.kwargs,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )

    return output


# Datasheet fp16-tensor-core peak (FP32 accumulate, dense) for the %-of-peak
# column. Set to the card you're on; None hides the column. L40 = 181 TFLOP/s.
PEAK_TFLOPS = 181.0


def run_autotune(A=4096, B=4096, C=4096, verbose=True):
    x = torch.randn((A, C), device=DEVICE, dtype=torch.float16)
    y = torch.randn((C, B), device=DEVICE, dtype=torch.float16)

    if verbose:
        print(f"autotuning over {len(get_autotune_configs())} configs for shape "
              f"A={A}, B={B}, C={C} (this runs every candidate config once)...")

    out = matmul_mnk(x, y)

    ref = torch.matmul(x, y)
    max_err = (out.float() - ref.float()).abs().max().item()

    best = tiled_matmul_mnk.best_config

    flop = 2 * A * B * C

    # cuBLAS baseline on the same fp16 matrices.
    torch_ms = triton.testing.do_bench(lambda: torch.matmul(x, y))
    torch_tflops = flop / (torch_ms * 1e-3) / 1e12

    ms = triton.testing.do_bench(lambda: matmul_mnk(x, y))
    tflops = flop / (ms * 1e-3) / 1e12

    if verbose:
        print(f"best config: {best}")
        print(f"max_err vs torch.matmul: {max_err:.4f}")
        print(f"torch.matmul: {torch_ms:.4f} ms | {torch_tflops:.2f} TFLOP/s")
        print(f"tuned kernel: {ms:.4f} ms | {tflops:.2f} TFLOP/s "
              f"({tflops / torch_tflops * 100:.1f}% of cuBLAS)")

    return {
        "shape": (A, B, C),
        "cublas_ms": torch_ms,
        "tuned_ms": ms,
        "cublas_tflops": torch_tflops,
        "tuned_tflops": tflops,
        "max_err": max_err,
        "best": best,
    }


def sweep_group_m(A=4096, B=4096, C=4096, group_ms=(1, 2, 4, 8, 16, 32)):
    """
    Sweep group_m values for L2 cache optimization
    """
    x = torch.randn((A, C), device=DEVICE, dtype=torch.float16)
    y = torch.randn((C, B), device=DEVICE, dtype=torch.float16)

    ref = torch.matmul(x, y)
    flop = 2 * A * B * C

    # One autotune pass to fix the tile shape; GROUP_M is then varied on top.
    matmul_mnk(x, y)
    base = tiled_matmul_mnk.best_config
    print(f"holding tile config fixed at: {base}")

    hdr = f"{'GROUP_M':>8} | {'time (ms)':>10} | {'TFLOP/s':>10} | {'vs G=1':>8} | {'max_err':>8}"
    print(hdr)
    print("-" * len(hdr))

    results = []
    baseline_tflops = None
    for group_m in group_ms:
        config = triton.Config(
            {**base.kwargs, "GROUP_M": group_m},
            num_stages=base.num_stages,
            num_warps=base.num_warps,
        )

        try:
            out = matmul_mnk(x, y, config=config)
        except triton.runtime.errors.OutOfResources as e:
            print(f"{group_m:>8} | skipped (out of resources: {e})")
            continue

        max_err = (out.float() - ref.float()).abs().max().item()

        ms = triton.testing.do_bench(lambda: matmul_mnk(x, y, config=config))
        tflops = flop / (ms * 1e-3) / 1e12
        if baseline_tflops is None:
            baseline_tflops = tflops
        speedup = tflops / baseline_tflops

        results.append((group_m, ms, tflops, max_err))
        print(f"{group_m:>8} | {ms:>10.4f} | {tflops:>10.2f} | {speedup:>7.2f}x | {max_err:>8.4f}")

    return results


def sweep_shapes(sizes=(4096, 8192, 16384)):
    """
    Sweep N values for square matrices
    """
    def pct_peak(t):
        return f"{t / PEAK_TFLOPS * 100:5.1f}%" if PEAK_TFLOPS else "   n/a"

    rows = []
    for n in sizes:
        # Re-tunes per shape (autotune key is A,B,C); prints its own progress.
        r = run_autotune(n, n, n, verbose=False)
        rows.append(r)
        print(f"  N={n:>6} tuned...  cuBLAS {r['cublas_ms']:.3f} ms / {r['cublas_tflops']:6.1f} TF  "
              f"tuned {r['tuned_ms']:.3f} ms / {r['tuned_tflops']:6.1f} TF")

    hdr = (f"{'N':>7} | {'cuBLAS ms':>10} | {'tuned ms':>10} | {'cuBLAS TF':>9} | {'tuned TF':>9} | "
           f"{'tuned/cuBLAS':>12} | {'cuBLAS %pk':>10} | {'tuned %pk':>10} | {'max_err':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        n = r["shape"][0]
        ratio = r["tuned_tflops"] / r["cublas_tflops"] * 100
        print(f"{n:>7} | {r['cublas_ms']:>10.4f} | {r['tuned_ms']:>10.4f} | "
              f"{r['cublas_tflops']:>9.1f} | {r['tuned_tflops']:>9.1f} | "
              f"{ratio:>11.1f}% | {pct_peak(r['cublas_tflops']):>10} | "
              f"{pct_peak(r['tuned_tflops']):>10} | {r['max_err']:>8.4f}")
    return rows


def print_cublas_kernel(A=4096, B=4096, C=4096, dtype=torch.float16):
    """
    Profile without ncu because lab machine doesn't have installed
    """
    from torch.profiler import profile, ProfilerActivity

    x = torch.randn((A, C), device=DEVICE, dtype=dtype)
    y = torch.randn((C, B), device=DEVICE, dtype=dtype)

    # Warmup so cuBLAS picks & caches its kernel for this exact shape.
    for _ in range(3):
        torch.matmul(x, y)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.matmul(x, y)
    torch.cuda.synchronize()

    print(f"cuBLAS kernel(s) for A={A}, B={B}, C={C}, dtype={dtype}:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=10, max_name_column_width=200))


def compare_at_config(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps,
                      A=4096, B=4096, C=4096, group_m=DEFAULT_GROUP_M):
    """
    Compare config against cuBLAS
    """
    smem = estimate_smem_bytes(BLOCK_M, BLOCK_N, BLOCK_K, num_stages)
    if smem > SIZE_SMEM:
        print(f"WARNING: config needs ~{smem/1024:.0f} KB SMEM but device has "
              f"{SIZE_SMEM/1024:.0f} KB - expect OutOfResources.")

    x = torch.randn((A, C), device=DEVICE, dtype=torch.float16)
    y = torch.randn((C, B), device=DEVICE, dtype=torch.float16)

    config = triton.Config(
        {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K, "GROUP_M": group_m},
        num_stages=num_stages,
        num_warps=num_warps,
    )

    ref = torch.matmul(x, y)
    flop = 2 * A * B * C

    try:
        out = matmul_mnk(x, y, config=config)
    except triton.runtime.errors.OutOfResources as e:
        print(f"config did not fit on the device: {e}")
        return None

    max_err = (out.float() - ref.float()).abs().max().item()

    ours_ms = triton.testing.do_bench(lambda: matmul_mnk(x, y, config=config))
    torch_ms = triton.testing.do_bench(lambda: torch.matmul(x, y))
    ours_tflops = flop / (ours_ms * 1e-3) / 1e12
    torch_tflops = flop / (torch_ms * 1e-3) / 1e12

    def pct(t):
        return f"{t / PEAK_TFLOPS * 100:5.1f}%" if PEAK_TFLOPS else "  n/a"

    print(f"config: BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K} "
          f"num_stages={num_stages} num_warps={num_warps} GROUP_M={group_m}  "
          f"(~{smem/1024:.0f} KB SMEM)")
    print(f"{'impl':>16} | {'time (ms)':>10} | {'TFLOP/s':>10} | {'%peak':>7}")
    print("-" * 52)
    print(f"{'ours @ config':>16} | {ours_ms:>10.4f} | {ours_tflops:>10.2f} | {pct(ours_tflops):>7}")
    print(f"{'torch (cuBLAS)':>16} | {torch_ms:>10.4f} | {torch_tflops:>10.2f} | {pct(torch_tflops):>7}")
    print(f"ours / cuBLAS: {ours_tflops / torch_tflops * 100:.1f}%   max_err: {max_err:.4f}")

    return {"ours_tflops": ours_tflops, "cublas_tflops": torch_tflops, "max_err": max_err}


if __name__ == "__main__":
    sweep_shapes()
    print()
    sweep_group_m()
    print()
    # ...then race our kernel pinned to that same tile (128x128x32, ~2 stages,
    # 8 warps, read off `ampere_..._128x128_..._stages_32x1`).
    N = [4096, 8192, 16384]
    for n in N:
        compare_at_config(128, 128, 32, A=n, B=n, C=n, num_stages=2, num_warps=8)
