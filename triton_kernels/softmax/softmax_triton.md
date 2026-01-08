# Softmax (Triton)

## Operator Overview

Softmax maps each row of a 2D tensor to a probability distribution:

- Input: `X` with shape `[M, N]`.
- Output: `Y` with shape `[M, N]`.
- Definition:
  - `Y[i, j] = exp(X[i, j] - max(X[i, :])) / sum_k exp(X[i, k] - max(X[i, :]))`

This stabilizes the exponentials by subtracting the row maximum.

## Triton Implementation

`triton_kernels/softmax/softmax.py` provides two kernels:

- `softmax_triton_kernel`: a three-pass implementation (row max, exp-sum, normalize+write).
- `online_softmax_triton_kernel`: a single-pass (streaming) reduction for max and sum, then a normalize+write pass.

### Kernel interface

Both kernels take:

- `input_ptr`: pointer to the input matrix.
- `output_ptr`: pointer to the output matrix.
- `row_stride`: number of columns (`N`).
- `BLOCK_SIZE`: number of columns processed per program instance.

A program instance corresponds to one row (`row_idx = tl.program_id(0)`), and it advances pointers by `row_idx * row_stride`.

### Three-pass softmax

For each row, the kernel does:

1) **Row max**
   - Loop over blocks of `BLOCK_SIZE` columns.
   - Load a block, apply a mask for tail elements, and compute the block max.
   - Reduce block maxima into a single `row_max`.

2) **Exp-sum**
   - Loop over blocks again.
   - Load, subtract `row_max`, and sum `exp` over each block.
   - Accumulate into `row_exp_sum`.

3) **Normalize + write**
   - Loop over blocks once more.
   - Load, subtract `row_max`, compute `exp / row_exp_sum`, and store to output.

### Online softmax (streaming max + sum)

The online kernel computes `row_max` and `row_exp_sum` in a streaming way:

- **Block 0** initializes `last_max` and `d` from the first block.
- For each subsequent block:
  - Compute `cur_max` for the new block.
  - Rescale the previous sum with `exp(last_max - cur_max)`.
  - Add the block sum of `exp(x - cur_max)`.
  - Update `last_max = cur_max`.
- After the reduction, a final pass writes normalized outputs.

## Online Softmax Derivation

Given a sequence of blocks, maintain:

- `m_k`: running maximum after processing block `k`.
- `d_k`: running sum of `exp(x - m_k)` after processing block `k`.

Initialization with block 0:

- `m_0 = max(x_0)`
- `d_0 = sum(exp(x_0 - m_0))`

Update with block `k`:

- `m_k = max(m_{k-1}, max(x_k))`
- `d_k = d_{k-1} * exp(m_{k-1} - m_k) + sum(exp(x_k - m_k))`

At the end, `m_k` is the row max and `d_k` is the sum of exponentials shifted by that max.

The output is:

- `y = exp(x - m_k) / d_k`

This matches the standard softmax definition while remaining numerically stable and avoiding a full two-pass reduction.
