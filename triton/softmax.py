
import torch
import triton
import triton.language as tl


# def _softmax_autotune_configs():
#     return [
#         triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
#         triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
#         triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
#         triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=3),
#         triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
#     ]

# @triton.autotune(
#     configs=_softmax_autotune_configs(),
#     key=["row_stride"],
# )
@triton.jit
def softmax_triton_kernel(
    input_ptr,
    output_ptr,
    row_stride,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    input_ptr = input_ptr + row_idx * row_stride
    output_ptr = output_ptr + row_idx + row_stride

    # === Reduction Pass ===
    # Pass 1: compute row max 
    row_max = -float('inf')
    for block_offset in range(0, row_stride, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        input_value = tl.load(input_ptr+col_indices, mask=col_indices<row_stride, other=0.0)
        row_max = tl.maximum(row_max, tl.max(input_value))

    # Pass 2: compute exp-sum
    row_exp_sum = 0
    for block_offset in range(0, row_stride, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        input_value = tl.load(input_ptr+col_indices, mask=col_indices<row_stride, other=0.0)
        input_value -= row_max
        row_exp_sum += tl.sum(tl.exp(input_value))
    
    # Pass 3: normalize + write
    for block_offset in range(0, row_stride, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        input_value = tl.load(input_ptr+col_indices, mask=col_indices<row_stride, other=0.0)
        input_value -= row_max
        output_value = tl.exp(input_value) / row_exp_sum
        tl.store(output_ptr+col_indices, output_value, mask=col_indices<row_stride)  

def softmax(
    input: torch.Tensor    
):
    output = torch.empty_like(input)
    n_rows, n_cols = input.shape
    softmax_triton_kernel[(n_rows)](
        input,
        output,
        n_cols,
    )


# @triton.autotune(
#     configs=_softmax_autotune_configs(),
#     key=["row_stride"],
# )
@triton.jit
def online_softmax_triton_kernel(
    input_ptr,
    output_ptr,
    row_stride,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    input_ptr = input_ptr + row_idx * row_stride
    output_ptr = output_ptr + row_idx * row_stride

    # block 0 
    col_indices = tl.arange(0, BLOCK_SIZE)
    input_value = tl.load(input_ptr+col_indices, mask=col_indices<row_stride, other=0.0)
    last_max = tl.max(input_value)
    d = tl.sum(tl.exp(input_value-last_max))
    
    # Pass 1: compute row max and exp-sum 
    for block_offset in range(1, row_stride, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        input_value = tl.load(input_ptr+col_indices, mask=col_indices<row_stride, other=0.0)
        cur_max = tl.maximum(last_max, tl.max(input_value))
        exp_a = tl.exp(last_max-cur_max)
        cur_exp = tl.exp(input_value - cur_max)
        cur_sum_exp = tl.sum(cur_exp)
        d = d * exp_a + cur_sum_exp
        last_max = cur_max

    for block_offset in range(0, row_stride, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        input_value = tl.load(input_ptr+col_indices, mask=col_indices<row_stride, other=0.0)
        exp_input = tl.exp(input_value - last_max)
        output_value = exp_input / d
        tl.store(output_ptr+col_indices, output_value, mask=col_indices<row_stride)  
    
def online_softmax(
    input: torch.Tensor    
):
    output = torch.empty_like(input)
    n_rows, n_cols = input.shape
    online_softmax_triton_kernel[(n_rows)](
        input,
        output,
        n_cols,
    )
