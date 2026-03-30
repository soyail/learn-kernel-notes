
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_triton_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,

    N,
    d,
    Br,
    Bc,
):
    rid = tl.program_id(0)
    cid = tl.program_id(1)

    offs_Q = rid * Br + tl.arange(0, Br)
    offs_K = cid * Bc + tl.arange(0, Bc)
    offs_V = cid * Bc + tl.arange(0, Bc)

    Q_ptrs = Q_ptr + offs_Q[]

    S = tl.empty([])
    for block_offset in range(0, N, Bc):
        col_indices = block_offset + tl.arange(0, Bc)
        K_block = tl.load(K_ptr)
        V_block = tl.load()

        for q_block_offset in range(0, N, Br):
            Q_block = tl.load(Q_ptr)
            S_block = tl.dot(Q_block, K_block)
            row_max = tl.max(S_block)
            exp = tl.exp(S_block - row_max)
            l = tl.sum(exp)

