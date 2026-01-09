
import torch
import triton
import triton.language as tl

@triton.jit
def gemm_triton_kernel(
    A_ptr, A_m_stride, A_k_stride,
    B_ptr, B_k_stride, B_n_stride,
    C_ptr, C_m_stride, C_n_stride,
    M, N, K,
    BLOCK_SIZE: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(A_m_stride, A_k_stride),
        offset=(pid_m * BLOCK_SIZE, 0),
        block_shape=(BLOCK_SIZE, BLOCK_SIZE),
        order=(1, 0),
    )

    B_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(B_k_stride, B_n_stride),
        offset=(0, pid_n * BLOCK_SIZE),
        block_shape=(BLOCK_SIZE, BLOCK_SIZE),
        order=(1, 0),
    )

    C_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(C_m_stride, C_n_stride),
        offset=(pid_m * BLOCK_SIZE, pid_n * BLOCK_SIZE),
        block_shape=(BLOCK_SIZE, BLOCK_SIZE),
        order=(1, 0),
    )

    c_acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for _ in range(0, K, BLOCK_SIZE):
        a = tl.load(A_block_ptr, boundary_check=(0,1), padding_option="zero")
        b = tl.load(B_block_ptr, boundary_check=(0,1), padding_option="zero")
        c_acc += tl.dot(a, b)

        A_block_ptr = A_block_ptr.advance((0, BLOCK_SIZE))
        B_block_ptr = B_block_ptr.advance((BLOCK_SIZE, 0))

    tl.store(pointer=C_block_ptr, value=c_acc, boundary_check=(0,1))


def gemm_triton(a: torch.Tensor, b: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("gemm_triton expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible shapes for matmul")

    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    grid = (triton.cdiv(m, block_size), triton.cdiv(n, block_size))
    gemm_triton_kernel[grid](
        a,
        a.stride(0),
        a.stride(1),
        b,
        b.stride(0),
        b.stride(1),
        c,
        c.stride(0),
        c.stride(1),
        m,
        n,
        k,
        BLOCK_SIZE=block_size,
    )
    return c
    
