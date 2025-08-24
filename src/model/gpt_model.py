import torch
import torch.nn as nn

from ..transformer import TransformerBlock, LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["ctx_len"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
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

# model = GPTModel(GPT_CONFIG_124M)

# import tiktoken
# import torch

# tokenizer = tiktoken.get_encoding("gpt2")
# batch = []

# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"

# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch = torch.stack(batch, dim=0)

# out = model(batch)
# print("Input batch:\n", batch)
# print("\nOutput shape:", out.shape)
# print(out)

# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}")

# # 计算总字节大小（假设每个参数均为占用4个字节的float32类型） 
# total_size_bytes = total_params * 4

# # 转换为兆字节（MB）
# total_size_mb = total_size_bytes / (1024 * 1024)

# print(f"Total size of the model: {total_size_mb:.2f} MB")