import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 因为要对权重矩阵按注意力头数进行拆分，所有输出维度必须是头数的整数倍
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # head_dim 就是拆分之后每个头应该输出的维度
        self.head_dim = d_out // num_heads 

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 形状为 (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 我们可以通过增加一个 num_heads 的维度来将矩阵分割到每个头
        # 维度变化: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置一下: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力权重
        # 基于矩阵乘法，简单地实现各个头的并行计算
        attn_scores = queries @ keys.transpose(2, 3) 
        # 一般来说我们会将掩码矩阵转化为 bool 值并基于序列的长度进行截断
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 需要将掩码矩阵 unsqueeze 两次，也就是增加两个维度，才能让掩码矩阵的维度和注意力权重对应上
        mask_unsqueezed = mask_bool.unsqueeze(0).unsqueeze(0)
        # 使用掩码矩阵来进行遮蔽
        attn_scores.masked_fill_(mask_unsqueezed, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 形状: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # 将多个头的输出重新组合回去 self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

# inputs = torch.tensor(
#   [[0.43, 0.15, 0.89], # Your     (x^1)
#    [0.55, 0.87, 0.66], # journey  (x^2)
#    [0.57, 0.85, 0.64], # starts   (x^3)
#    [0.22, 0.58, 0.33], # with     (x^4)
#    [0.77, 0.25, 0.10], # one      (x^5)
#    [0.05, 0.80, 0.55]] # step     (x^6)
# )

# batch = torch.stack((inputs, inputs), dim=0)

# # 试验一下
# torch.manual_seed(123)

# batch_size, block_size, d_in = batch.shape
# d_out = 4
# mha = MultiHeadAttention(d_in, d_out, block_size, 0.0, num_heads=2)

# context_vecs = mha(batch)

# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)