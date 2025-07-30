import torch
import torch.nn as nn

class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, block_size, dropout, qkv_bias=False):
        '''
        构造函数，输入参数如下：
        d_in: 输入的维度
        d_out: 输出的维度
        block_size: 注意力权重矩阵的大小
        dropout: dropout 比例
        qkv_bias: 是否对 query、key 和 value 加偏置
        '''
        super().__init__()
        self.d_out = d_out
        # 根据前文，每一个权重矩阵都是 d_in x d_out 的线性层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 一个 dropout 层
        self.dropout = nn.Dropout(dropout) 
        # 一个掩码矩阵，下三角为 1，其余为 0
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1)) # New

    def forward(self, x):
        '''
        前向传播函数，输入参数为 x，维度为 b x num_tokens x d_in，输出维度为 b x num_tokens x d_out
        '''
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # transpose 是为了实现矩阵乘法
        attn_scores = queries @ keys.transpose(1, 2)
        # 即上文说过的，将掩码从 0 修改为 -inf，再进行遮蔽操作
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # 经过 softmax 
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
        # 进行 dropout
        attn_weights = self.dropout(attn_weights) # New
        # 得到最后结果
        context_vec = attn_weights @ values
        return context_vec

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

torch.manual_seed(123)

batch = torch.stack((inputs, inputs), dim=0)
block_size = batch.shape[1]
ca = CausalAttention(3, 2, block_size, 0.0)

context_vecs = ca(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)