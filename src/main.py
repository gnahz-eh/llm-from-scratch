import re
import tiktoken
import torch
import os
import urllib.request
from importlib.metadata import version

from .model import GPTModel
from .utils import *
import urllib.request

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




#---------------------------------------------------- CH04-----------------------------------------------------
# tokenizer = tiktoken.get_encoding("gpt2")

# start_context = "Hello, I am"
# encoded = tokenizer.encode(start_context)
# print("encoded:", encoded)
# encoded_tensor = torch.tensor(encoded).unsqueeze(0) #A
# print("encoded_tensor.shape:", encoded_tensor.shape)

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

# model.eval() # 关闭 dropout

# out = generate_text_simple(
#     model=model,
#     idx=encoded_tensor, 
#     max_new_tokens=6, 
#     context_size=GPT_CONFIG_124M["ctx_len"]
# )

# print("Output:", out)
# print("Output length:", len(out[0]))

# decoded_text = tokenizer.decode(out.squeeze(0).tolist())
# print(decoded_text)

#---------------------------------------------------- CH05 -----------------------------------------------------
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "ctx_len": 256,       # Shortened context length (orig: 1024)
    "emb_dim": 768,       # Embedding dimension
    "n_heads": 12,        # Number of attention heads
    "n_layers": 12,       # Number of layers
    "drop_rate": 0.1,     # Dropout rate
    "qkv_bias": False     # Query-key-value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval();  # Disable dropout during inference

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["ctx_len"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [588,  428,  11311]]) #  " really like chocolate"]

with torch.no_grad():
    logits = model(inputs)
print(logits) # Shape: (batch_size, num_tokens, vocab_size)
probas = torch.softmax(logits, dim=-1) # 词表中每个标记的预测概率
print(probas) # Shape: (batch_size, num_tokens, vocab_size)
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

with open("src/resources/the-verdict.txt", "r", encoding="utf-8") as f:
    text_data = f.read()
# First 100 characters
print(text_data[:99])
# Last 100 characters
print(text_data[-99:])

total_char = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print("Characters:", total_char)
print("Tokens:", total_tokens)

# 训练集/验证集数据比
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# 训练集/验证集数据比
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["ctx_len"],
    stride=GPT_CONFIG_124M["ctx_len"],
    drop_last=True,
    shuffle=True
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["ctx_len"],
    stride=GPT_CONFIG_124M["ctx_len"],
    drop_last=False,
    shuffle=False
)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel() # 使用numel()函数统计一个batch中的token数量

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # 对于nn.Module类的模型，不需要执行model = model.to(device)这样的赋值操作。


torch.manual_seed(123) # 出于代码结果的可复现性的考虑，显式地设定manual_seed
train_loss = calc_loss_loader(train_loader, model, device)
val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# -----------------------------TRAINING CODE--------------------------------------
# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

# num_epochs = 10
# train_losses, val_losses, tokens_seen = train_model_simple(
#     model, train_loader, val_loader, optimizer, device,
#     num_epochs=num_epochs, eval_freq=5, eval_iter=1,
#     start_context="Every effort moves you", tokenizer=tokenizer
# )

# --------------------------------- GENERATION WITH TEMPERATURE SCALING AND TOP K -------------------------------------------
# torch.manual_seed(123)

# token_ids = generate(
#     model=model,
#     idx=text_to_token_ids("Every effort moves you", tokenizer),
#     max_new_tokens=15,
#     context_size=GPT_CONFIG_124M["ctx_len"],
#     top_k=25,
#     temperature=1.4
# )

# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
#-------------------------------------------------------------------
try:
    print("TensorFlow version:", version("tensorflow"))
except Exception:
    print("TensorFlow version: Not installed")
print("tqdm version:", version("tqdm"))


settings, params, _ = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
print("Settings:", settings)
print("Parameters dictionary keys:", params.keys())

print(params["wte"]) 
print("Token embedding weight tensor dimensions:", params["wte"].shape)

# --------------------------------- LOAD GPT2 MODEL AND GENERATE TEXT -------------------------------------------
# 将模型配置参数定义在一个字典中
model_configs = {
    "gpt2-small": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 复制基础配置，并使用特定的模型设置进行更新
model_name = "gpt2-small"  # Example model name
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"ctx_len": 1024, "qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval();

load_weights_into_gpt(gpt, params)
gpt.to(device);

torch.manual_seed(123)

token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("how about the name of ZhuYue", tokenizer),
    max_new_tokens=250,
    context_size=NEW_CONFIG["ctx_len"],
    top_k=50,
    temperature=1.5
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))