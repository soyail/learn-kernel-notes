import os
import torch
import triton

from gemm import gemm_triton


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["K"],
        x_vals=[128 * i for i in range(2, 32)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="TFLOPS",
        plot_name="gemm-performance",
        args={"M": 2048, "N": 2048, "dtype": torch.float16},
    )
)
def benchmark(N, M, K, dtype, provider):
    a = torch.randn((M, K), dtype=dtype, device=torch.device("cuda:0"))
    b = torch.randn((K, N), dtype=dtype, device=torch.device("cuda:0"))
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: gemm_triton(a, b), quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles
        )

    def tflops(ms_val):
        flops = 2 * M * N * K
        return flops * 1e-12 / (ms_val * 1e-3)

    return tflops(ms), tflops(max_ms), tflops(min_ms)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "benchmarks")
os.makedirs(OUTPUT_DIR, exist_ok=True)
benchmark.run(show_plots=True, print_data=True, save_path=OUTPUT_DIR)
