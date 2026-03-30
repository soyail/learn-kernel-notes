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

        B = self.block_size

        # TODO your code here
        scale_factor = D ** (-0.5)
        out = []
        
        for i in range(n_blocks):
            q_i = q[:, i*B:(i+1)*B, :]
            # online softmax三元组 
            # m: 计算过的block中的最大值
            # l: 当前累计的exp和
            # o: 分子
            m = torch.full((H,B,1), float('-inf'), device=q.device)
            l = torch.zeros((H,B,1), device=q.device)
            o = torch.zeros((H,B,D),device=q.device)
            for j in range(n_blocks):
                if block_mask[i][j]==0:
                    continue
                k_j = k[:, j*B:(j+1)*B, :] # 【H, B, D】
                v_j = v[:, j*B:(j+1)*B, :] # [H, B, D]

                logits = torch.bmm(q_i, k_j.transpose(1,2)) * scale_factor # o_l: [H, B, B]
                # e^(x_i-d_max)
                m_new = logits.max(dim=-1, keep_dim=True).values # [H, B, 1]
                m_new = torch.max(m, m_new)
                scale_old = torch.exp(m - m_new)  # [H,B,1]
                scale_new = torch.exp(logits - m_new) #[H,B,B]
                l = scale_old * l + scale_new.sum(dim=-1) # [H,B,1]
                o = scale_old * o + torch.bmm(scale_new, v_j) # logits: [H,B,B] v_j [H,B,D]

                m = m_new
            out.append(o/l)

        return torch.cat(out)
    
if __name__ == "__main__":
    torch.manual_seed(19260817)
    H = 4
    N = 32 * 1024
    D = 128
    BLOCK_SIZE = 256
    device = "cuda"
    print(f"H={H}, N={N}, D={D}, BLOCK_SIZE={BLOCK_SIZE}")

    q = torch.randn(H, N, D, dtype=torch.float32)
    k = torch.randn(H, N, D, dtype=torch.float32)
    v = torch.randn(H, N, D, dtype=torch.float32)

    num_blocks = N // BLOCK_SIZE
    raw_block_mask = torch.randint(0, 2, (num_blocks, num_blocks))
    for i in range(num_blocks):
        raw_block_mask[i][i] = 1

    # Time the block sparse attention
    start_ts = time.time()
    model = BlockSparseAttention(H, D, block_size=BLOCK_SIZE)
    output = model(q, k, v, raw_block_mask)
    end_ts = time.time()
    print(f"Time: {end_ts - start_ts:.3f} seconds")

    # Dense attention for comparison
    dense_mask = raw_block_mask.repeat_interleave(BLOCK_SIZE, dim=0).repeat_interleave(BLOCK_SIZE, dim=1).float()
    dense_mask = torch.triu(dense_mask, diagonal=0).bool()
    ref_out = F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask)

    # Cosine similarity between outputs
    cos_sim = F.cosine_similarity(output, ref_out, dim=-1).mean()
    ref_out = ref_out.transpose(0, 1).reshape(-1, H * D)
    output = output.transpose(0, 1).reshape(-1, H * D)
    print(f"Cos sim: {cos_sim}")