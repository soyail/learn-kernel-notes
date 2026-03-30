import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockSparseAttention(nn.Module):
    def __init__(self, n_heads=4, head_dim=64, block_size=32):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.block_size = block_size

    def forward(self, q, k, v, block_mask):
        """
        Args:
            q, k, v: [n_heads, seq_len, head_dim], FP32
            block_mask: [seq_len // block_size, seq_len // block_size], INT32, 1=valid, 0=masked
        Returns:
            out: [n_heads, seq_len, head_dim]
        """
        H, N, D = q.shape
        assert N % self.block_size == 0, "seq_len must be divisible by block_size"
        n_blocks = N // self.block_size

        # TODO: Your code here.
        B = self.block_size
        out = [] # out: [H, N, N]
        scale = D ** (-0.5)
        for i in range(n_blocks):
            q_i = q[:, i*B:(i+1)*B, :] # [H, B, D]
            # online softmax l,m,o
            m = torch.full((H,B,1), float('-inf'), device=q.device) # m: [H,B,1]
            l = torch.zeros((H,B,1), dtype=torch.float32, device=q.device) # l: [H,B,1]
            o = torch.zeros((H,B,D), dtype=torch.float32, device=q.device) # o: [H,B,D]

            for j in range(n_blocks):
                if block_mask[i][j] == 0:
                    continue
                k_j = k[:, j*B:(j+1)*B, :] # [H, B, D]
                v_j = v[:, j*B:(j+1)*B, :] # [H, B, D]
                logits = torch.bmm(q_i, k_j.transpose(1,2)) * scale # logits: [H, B, B]
                m_new = logits.max(dim=-1, keepdim=True).values #[H,B,1]
                m_new = torch.maximum(m_new, m)
                # l: e^(x_i - m)
                scale_old = torch.exp(m - m_new)
                scale_new = torch.exp(logits - m_new) # 
                # e^(x_i-m) * e^{m-m_new} + \sum_j {x_j-m_new}

                l = scale_old * l + scale_new.sum(dim=-1, keepdim=True) 
                o = scale_old * o + torch.bmm(scale_new, v_j)
                m = m_new

            out.append(o/l) # [H, n_blocks*B , B]

        out = torch.cat(out, dim=1)
        return out


if __name__ == "__main__":
    torch.manual_seed(19260817)

    H = 4
    N = 32 * 1024  # 32k
    D = 128
    BLOCK_SIZE = 256

    device = "cuda"

    print(f"{H=}, {N=}, {D=}, {BLOCK_SIZE=}, {device=}")

    q = torch.randn(H, N, D, dtype=torch.float32).to(device)
    k = torch.randn(H, N, D, dtype=torch.float32).to(device)
    v = torch.randn(H, N, D, dtype=torch.float32).to(device)

    num_blocks = N // BLOCK_SIZE
    raw_block_mask = torch.randint(-3, 2, (num_blocks, num_blocks)).clamp(0, 1)
    for i in range(num_blocks):
        raw_block_mask[i, i] = 1
    print(f"Sparsity: {100 * (1 - raw_block_mask.sum().item() / raw_block_mask.nelement())}%")
    raw_block_mask = raw_block_mask.to(device)

    start_ts = time.time()
    model = BlockSparseAttention(H, D, block_size=BLOCK_SIZE)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    
    start_event.record()
    custom_out = model(q, k, v, raw_block_mask)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Time: {elapsed_time_ms / 1000:.6f} seconds")

    dense_mask = raw_block_mask.repeat_interleave(BLOCK_SIZE, dim=0).repeat_interleave(BLOCK_SIZE, dim=1).float()
    sdpa_mask = torch.zeros_like(dense_mask).masked_fill(dense_mask == 0, float('-inf')).to(device)
    ref_out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)

    cos_sim = F.cosine_similarity(
        custom_out.transpose(0, 1).reshape(-1, H * D),
        ref_out.transpose(0, 1).reshape(-1, H * D),
    )
    print(f"{cos_sim=}")

    min_cos_sim = cos_sim.min()
    print(f"{min_cos_sim=}")

    if min_cos_sim > 0.999:
        print("\n✅ PASSED")
    else:
        print("\n❌ FAILED")