import pytest
import torch

from softmax import softmax as softmax_triton
from softmax import online_softmax as online_softmax_triton


def get_tol(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return dict(atol=1e-2, rtol=1e-2)
    if dtype == torch.float16:
        return dict(atol=1e-3, rtol=1e-3)
    return dict(atol=1e-5, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("m", [128, 1024])
@pytest.mark.parametrize("n", [128, 512, 1024, 2048])
def test_triton_softmax(dtype: torch.dtype, m: int, n: int) -> None:
    device = torch.device("cuda:0")
    x = torch.randn((m, n), dtype=dtype, device=device)

    y_triton = softmax_triton(x)
    y_ref = torch.nn.functional.softmax(x, dim=-1)

    torch.testing.assert_close(y_triton, y_ref, **get_tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("m", [128, 1024])
@pytest.mark.parametrize("n", [127, 512, 1024, 2048])
def test_triton_online_softmax(dtype: torch.dtype, m: int, n: int) -> None:
    device = torch.device("cuda:0")
    x = torch.randn((m, n), dtype=dtype, device=device)

    y_triton = online_softmax_triton(x)
    y_ref = torch.nn.functional.softmax(x, dim=-1)

    torch.testing.assert_close(y_triton, y_ref, **get_tol(dtype))
