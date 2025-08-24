import torch
import torch.nn as nn

from ..attention import MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            block_size=cfg["ctx_len"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 注意力块中的Shortcut连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)
        x = x + shortcut  # 与原始输入块求和

        # 前馈块中的Shortcut连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # 与原始输入块求和

        return x
    
# GPT_CONFIG_124M = {
#     "vocab_size": 50257,  # 词表大小
#     "ctx_len": 1024,      # 上下文长度
#     "emb_dim": 768,       # 嵌入维度
#     "n_heads": 12,        # 注意力头（attention heads）的数量
#     "n_layers": 12,       # 模型层数
#     "drop_rate": 0.1,     # Dropout rate
#     "qkv_bias": False     # Query-Key-Value bias
# }

# torch.manual_seed(123)

# x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
# block = TransformerBlock(GPT_CONFIG_124M)
# output = block(x)

# print("Input shape:", x.shape)
# print("Output shape:", output.shape)