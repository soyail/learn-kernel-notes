import torch
import triton

from softmax import softmax as softmax_triton
from softmax import online_softmax as online_softmax_triton


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 32)],
        line_arg="provider",
        line_vals=["triton", "triton_online", "torch"],
        line_names=["Triton", "Triton Online", "Torch"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 4096, "dtype": torch.float16},
    )
)
def benchmark(N, M, dtype, provider):
    x = torch.randn((M, N), dtype=dtype, device=torch.device("cuda:0"))
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: softmax_triton(x), quantiles=quantiles
        )
    elif provider == "triton_online":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: online_softmax_triton(x), quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.softmax(x, dim=-1), quantiles=quantiles
        )

    def gbps(ms_val):
        return 2 * x.nelement() * x.element_size() * 1e-9 / (ms_val * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, print_data=True)
