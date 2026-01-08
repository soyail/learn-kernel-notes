#!/usr/bin/env python3

import sys
import torch
import click
from loguru import logger

from triton_kernels.rmsnorm import rmsnorm_triton, fused_add_rmsnorm_triton


def _rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float):
    x_fp32 = x.float()
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    y = x_fp32 * torch.rsqrt(var + eps) * weight.float()
    return y.to(dtype=x.dtype)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor):
    if actual.dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def benchmark_rmsnorm(
    m,
    n,
    dtype=torch.float16,
    warmup_iters=10,
    bench_iters=100,
    backend="triton",
    fused_add=False,
    eps=1e-6,
):
    logger.info(
        f"Benchmarking {backend} rmsnorm with M={m}, N={n}, dtype={dtype}, fused_add={fused_add}"
    )

    x = torch.randn(m, n, dtype=dtype, device=DEVICE)
    weight = torch.randn(n, dtype=dtype, device=DEVICE)
    residual = torch.randn(m, n, dtype=dtype, device=DEVICE) if fused_add else None

    if backend == "triton":
        if fused_add:
            rmsnorm_fn = lambda x, w, r: fused_add_rmsnorm_triton(x, r, w, eps=eps)
        else:
            rmsnorm_fn = lambda x, w, r: rmsnorm_triton(x, w, eps=eps)
    elif backend == "torch":
        if fused_add:
            rmsnorm_fn = lambda x, w, r: _rmsnorm_reference(x + r, w, eps)
        else:
            rmsnorm_fn = lambda x, w, r: _rmsnorm_reference(x, w, eps)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    logger.info(f"Warming up for {warmup_iters} iterations...")
    for _ in range(warmup_iters):
        _ = rmsnorm_fn(x, weight, residual)
    torch.cuda.synchronize()

    logger.info(f"Running benchmark for {bench_iters} iterations...")
    times_ms = []

    for _ in range(bench_iters):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        output = rmsnorm_fn(x, weight, residual)
        end_event.record()

        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    times_ms.sort()
    n_remove = int(0.1 * len(times_ms))
    if n_remove > 0:
        trimmed_times = times_ms[n_remove:-n_remove]
        logger.info(f"Trimmed {n_remove} outliers from each end ({2*n_remove} total)")
    else:
        trimmed_times = times_ms

    avg_time_ms = sum(trimmed_times) / len(trimmed_times)

    num_elements = m * n
    if fused_add:
        bytes_accessed = 3 * num_elements * x.element_size() + n * weight.element_size()
    else:
        bytes_accessed = 2 * num_elements * x.element_size() + n * weight.element_size()
    bandwidth_gbps = (bytes_accessed * 1e-9) / (avg_time_ms * 1e-3)

    logger.success("Benchmark completed:")
    logger.info(f"  Average time per iteration: {avg_time_ms:.4f} ms")
    logger.info(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
    logger.info(f"  Total elements: {num_elements:,}")
    logger.info(f"  Memory accessed: {bytes_accessed / 1e9:.2f} GB")

    return output


@click.command()
@click.option("--M", type=int, required=True, help="Number of rows (batch dimension)")
@click.option("--N", type=int, required=True, help="Number of columns (feature dimension)")
@click.option(
    "--backend",
    type=click.Choice(["triton", "torch"]),
    default="triton",
    help="Backend to use: triton (custom kernel) or torch (PyTorch)",
)
@click.option(
    "--dtype",
    type=click.Choice(["float16", "float32"]),
    default="float16",
    help="Data type for tensors",
)
@click.option("--warmup", type=int, default=10, help="Number of warmup iterations")
@click.option("--iters", type=int, default=100, help="Number of benchmark iterations")
@click.option("--fused-add", is_flag=True, help="Benchmark fused add + rmsnorm kernel")
@click.option(
    "--profile-only", is_flag=True, help="Run only a single iteration for profiling"
)
@click.option(
    "--compare",
    is_flag=True,
    help="Compare against PyTorch RMSNorm reference for correctness",
)
def main(m, n, backend, dtype, warmup, iters, fused_add, profile_only, compare):
    """Benchmark rmsnorm kernel for ncu profiling."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a CUDA-capable GPU.")
        sys.exit(1)

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    dtype_tensor = dtype_map[dtype]

    logger.info("Starting rmsnorm benchmark with parameters:")
    logger.info(f"  Backend: {backend}")
    logger.info(f"  M (rows): {m}")
    logger.info(f"  N (cols): {n}")
    logger.info(f"  dtype: {dtype}")
    logger.info(f"  warmup iterations: {warmup}")
    logger.info(f"  benchmark iterations: {iters}")
    logger.info(f"  fused add: {fused_add}")

    x = torch.randn(m, n, dtype=dtype_tensor, device=DEVICE)
    weight = torch.randn(n, dtype=dtype_tensor, device=DEVICE)
    residual = torch.randn(m, n, dtype=dtype_tensor, device=DEVICE) if fused_add else None

    if profile_only:
        logger.info("Profile-only mode: running single iteration for ncu profiling")
        if backend == "triton":
            if fused_add:
                output = fused_add_rmsnorm_triton(x, residual, weight)
            else:
                output = rmsnorm_triton(x, weight)
        else:
            if fused_add:
                output = _rmsnorm_reference(x + residual, weight, 1e-6)
            else:
                output = _rmsnorm_reference(x, weight, 1e-6)
        torch.cuda.synchronize()
        logger.success("Single iteration completed for profiling")
    else:
        output = benchmark_rmsnorm(
            m,
            n,
            dtype_tensor,
            warmup,
            iters,
            backend,
            fused_add=fused_add,
        )

    if compare and backend == "triton":
        if fused_add:
            expected = _rmsnorm_reference(x + residual, weight, 1e-6)
        else:
            expected = _rmsnorm_reference(x, weight, 1e-6)
        _assert_close(output, expected)
        logger.success("Correctness check passed against PyTorch reference")

    logger.success("Benchmark completed successfully!")


if __name__ == "__main__":
    DEVICE = torch.device("cuda")
    main()
