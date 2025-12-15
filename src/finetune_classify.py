# Direct import
from src.finetune import (
    download_and_unzip_spam_data,
    create_balanced_dataset,
    random_split,
    SpamDataset,
)

# Or module import
import urllib.request
import zipfile
import os
from pathlib import Path

# Import required modules
import pandas as pd
import tiktoken
from torch.utils.data import DataLoader
import torch

# Use the existing file in src/resources directory
data_file_path = "./src/resources/sms_spam_collection/SMSSpamCollection.tsv"

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
print(df)


balanced_df = create_balanced_dataset(df)
print("--->")
print(balanced_df["Label"].value_counts())


balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
# 测试大小默认为 0.2

data_output_path = "./src/resources/sms_spam_collection/processed"

train_df.to_csv(data_output_path + "/train.csv", index=None)
validation_df.to_csv(data_output_path + "/validation.csv", index=None)
test_df.to_csv(data_output_path + "/test.csv", index=None)

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

train_dataset = SpamDataset(
    csv_file=data_output_path + "/train.csv", max_length=None, tokenizer=tokenizer
)

print(train_dataset.max_length)

val_dataset = SpamDataset(
    csv_file=data_output_path + "/validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer,
)
test_dataset = SpamDataset(
    csv_file=data_output_path + "/test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer,
)

num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

print("Train loader:")
for input_batch, target_batch in train_loader:
    pass

print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)

print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")


CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
# settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

# model = GPTModel(BASE_CONFIG)
# load_weights_into_gpt(model, params)
# model.eval();