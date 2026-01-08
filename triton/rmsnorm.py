
import torch
import triton
import triton.language as tl

#TODO add triton autotune

@triton.jit
def rmsnorm_triton_kernel(
    input_ptr,
    output_ptr,
    weight_ptr,
    feature_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    input_ptr += row_idx * feature_dim
    output_ptr += row_idx * feature_dim

    sum_of_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_offset in range(0, feature_dim, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        input_value = tl.load(input_ptr+col_indices, mask=col_indices<feature_dim, other=0.0).to(tl.float32)
        sum_of_squares += input_value * input_value

    variance = tl.sum(sum_of_squares) / feature_dim
    reciprocal_std = 1 / tl.sqrt(variance + eps)
    
    for block_offset in range(0, feature_dim, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        input_value = tl.load(input_ptr+col_indices, mask=col_indices<feature_dim, other=0.0).to(tl.float32)
        weight_value = tl.load(weight_ptr+col_indices, mask=col_indices<feature_dim, other=0.0)
        normalized_values = input_value * reciprocal_std 
        output_value = normalized_values * weight_value
        tl.store(output_ptr+col_indices, output_value, mask=col_indices<feature_dim)

def rmsnorm_triton(
    input: torch.Tensor, weight: torch.Tensor, eps=1e-6
):
    output = torch.empty_like(input)
    n_rows, feature_dim = input.shape()
    BLOCK_SIZE = 1024
    rmsnorm_triton_kernel[(n_rows)](
        input,
        output,
        weight,
        feature_dim,
        eps,
        BLOCK_SIZE,
    )

@triton.jit
def fused_add_rmsnorm_triton_kernel(
    input_ptr,
    residual_ptr,
    output_ptr,
    weight_ptr,
    feature_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    input_ptr += row_idx * feature_dim
    residual_ptr += row_idx * feature_dim
    output_ptr += row_idx * feature_dim

    sum_of_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_offset in range(0, feature_dim, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        input_value = tl.load(input_ptr+col_indices, mask=col_indices<feature_dim, other=0.0).to(tl.float32)
        residual_value = tl.load(residual_ptr+col_indices, mask=col_indices<feature_dim, other=0.0).to(tl.float32)
        input_value += residual_value
        sum_of_squares += input_value * input_value
        tl.store(input_ptr+col_indices, input_value, mask=col_indices<feature_dim)

    variance = tl.sum(sum_of_squares) / feature_dim
    reciprocal_std = 1 / tl.sqrt(variance + eps)
    
    for block_offset in range(0, feature_dim, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        input_value = tl.load(input_ptr+col_indices, mask=col_indices<feature_dim, other=0.0).to(tl.float32)
        weight_value = tl.load(weight_ptr+col_indices, mask=col_indices<feature_dim, other=0.0)
        normalized_values = input_value * reciprocal_std 
        output_value = normalized_values * weight_value
        tl.store(output_ptr+col_indices, output_value, mask=col_indices<feature_dim)

def fused_add_rmsnorm_triton(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps=1e-6
):
    output = torch.empty_like(input)
    n_rows, feature_dim = input.shape()
    BLOCK_SIZE = 1024
    fused_add_rmsnorm_triton_kernel[(n_rows)](
        input,
        residual,
        output,
        weight,
        feature_dim,
        eps,
        BLOCK_SIZE,
    )