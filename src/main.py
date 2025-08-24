import re
import tiktoken
import torch

from .model import GPTModel
from .utils import generate_text_simple

# from tokenizer.simple_tokenizer import SimpleTokenizerV2

# 1. prepare the tokens from the traning data
# with open("src/resources/the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
# preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
# preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(len(preprocessed))
# print(preprocessed[:30])

# # 6. add dedicated tokens
# all_tokens = sorted(list(set(preprocessed)))
# all_tokens.extend(["<|endoftext|>", "<|unk|>"])
# vocab = {token:integer for integer,token in enumerate(all_tokens)}
# print(len(vocab.items()))
# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)

# # 8. test the new tokenizer
# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text1, text2))
# print(text)

# tokenizer = SimpleTokenizerV2(vocab)
# print(tokenizer.encode(text))
# print(tokenizer.decode(tokenizer.encode(text)))





tokenizer = tiktoken.get_encoding("gpt2")

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) #A
print("encoded_tensor.shape:", encoded_tensor.shape)

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # 词表大小
    "ctx_len": 1024,      # 上下文长度
    "emb_dim": 768,       # 嵌入维度
    "n_heads": 12,        # 注意力头（attention heads）的数量
    "n_layers": 12,       # 模型层数
    "drop_rate": 0.1,     # Dropout rate
    "qkv_bias": False     # Query-Key-Value bias
}

torch.manual_seed(123)

model = GPTModel(GPT_CONFIG_124M)

model.eval() # 关闭 dropout

out = generate_text_simple(
    model=model,
    idx=encoded_tensor, 
    max_new_tokens=6, 
    context_size=GPT_CONFIG_124M["ctx_len"]
)

print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)