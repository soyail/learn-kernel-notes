# Matmul (Triton)

## Operator Overview

Matrix multiplication computes:

- Inputs: `A` with shape `[M, K]`, `B` with shape `[K, N]`.
- Output: `C` with shape `[M, N]`.
- Definition: `C[i, j] = sum_k A[i, k] * B[k, j]`.

## Triton Implementation Procedures

`triton_kernels/matmul/matmul.py` provides two implementations:

- `matmul_v0`: a straightforward blocked matmul using `make_block_ptr` for A/B/C tiles.
- `matmul_v1`: a tiled kernel using explicit pointer arithmetic, program-id swizzle, and per-tile K loop.

### Common structure

Both kernels use tiling so each program instance computes one output tile:

1) Map program IDs to output tiles (`pid_m`, `pid_n`).
2) Build pointers to A and B sub-tiles for the current output tile.
3) Loop over K in blocks, accumulating into an fp32 accumulator.
4) Store the tile back to C with an out-of-bounds mask.

### matmul_v0

`matmul_v0_kernel` uses `tl.make_block_ptr` for block pointers:

- `block_shape=(BLOCK_SIZE, BLOCK_SIZE)` for A/B/C tiles.
- Loads with boundary checks and `padding_option="zero"` to handle edge tiles.
- Accumulates into `c_acc` and stores the fp16 output tile.

This is the baseline implementation for correctness and clarity.

## matmul_v1 Main Optimization and How It Works

`matmul_v1_kernel` implements a more optimized tiling strategy:

1) **Program-id swizzle with `GROUP_SIZE_M`**
   - Instead of a 2D grid, it uses a 1D grid and remaps `pid` into `(pid_m, pid_n)`.
   - Tiles are grouped along M so blocks within a group share the same N range.
   - This improves L2 cache reuse because nearby blocks re-read similar B tiles.

2) **Explicit pointer arithmetic**
   - `a_ptrs` points to a `[BLOCK_SIZE_M, BLOCK_SIZE_K]` tile of A.
   - `b_ptrs` points to a `[BLOCK_SIZE_K, BLOCK_SIZE_N]` tile of B.
   - Pointers are advanced each K-iteration (`a_ptrs += BLOCK_SIZE_K * stride_ak`).

3) **K-tiled accumulation**
   - The kernel loops over `K` in `BLOCK_SIZE_K` chunks.
   - Each iteration loads A/B sub-tiles with masks for tail handling.
   - Accumulates using `tl.dot(a, b)` into an fp32 `accumulator`.

4) **Contiguity hints for vectorization**
   - `tl.multiple_of` + `tl.max_contiguous` on `offs_am` and `offs_bn` tell the compiler the access pattern is aligned and contiguous.
   - This helps Triton generate vectorized loads.

5) **Output store with masking**
   - The final tile is stored with a mask to guard out-of-bounds elements.

In short, `matmul_v1` improves performance by reducing memory traffic with tile reuse, improving cache locality via swizzle, and enabling better vectorization through contiguity hints.
