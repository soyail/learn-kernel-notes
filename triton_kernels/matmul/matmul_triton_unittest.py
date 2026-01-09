import pytest
import torch

from gemm import matmul_v0, matmul_v1


def get_tol(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return dict(atol=1e-2, rtol=1e-2)
    if dtype == torch.float16:
        return dict(atol=1e-2, rtol=1e-2)
    return dict(atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("m,n,k", [(128, 128, 128), (256, 128, 192), (257, 129, 130)])
def test_triton_gemm(dtype: torch.dtype, m: int, n: int, k: int) -> None:
    device = torch.device("cuda:0")
    a = torch.randn((m, k), dtype=dtype, device=device)
    b = torch.randn((k, n), dtype=dtype, device=device)

    y_triton = matmul_v0(a, b)
    y_ref = torch.matmul(a, b)

    torch.testing.assert_close(y_triton, y_ref, **get_tol(dtype))

@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("m,n,k", [(128, 128, 128), (256, 128, 192), (257, 129, 130)])
def test_triton_matmul(dtype: torch.dtype, m: int, n: int, k: int) -> None:
    device = torch.device("cuda:0")
    a = torch.randn((m, k), dtype=dtype, device=device)
    b = torch.randn((k, n), dtype=dtype, device=device)

    y_triton = matmul_v1(a, b)
    y_ref = torch.matmul(a, b)

    torch.testing.assert_close(y_triton, y_ref, **get_tol(dtype))
