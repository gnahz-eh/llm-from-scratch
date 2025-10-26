"""
Main script for GPT model training and text generation.
This script demonstrates the complete pipeline from data preparation to text generation.
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import torch
import tiktoken
from importlib.metadata import version

from .model import GPTModel
from .utils import *

# ============================================================================
# 2. CONFIGURATION
# ============================================================================
# Base GPT configuration (124M parameter model)
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "ctx_len": 256,       # Context length (shortened from 1024 for demo)
    "emb_dim": 768,       # Embedding dimension
    "n_heads": 12,        # Number of attention heads
    "n_layers": 12,       # Number of layers
    "drop_rate": 0.1,     # Dropout rate
    "qkv_bias": False     # Query-key-value bias
}

# Model variants configuration
MODEL_CONFIGS = {
    "gpt2-small": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# ============================================================================
# 3. DEVICE SETUP AND SEEDING
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)  # For reproducibility

# ============================================================================
# 4. INITIAL MODEL TESTING (UNTRAINED)
# ============================================================================
print("=" * 60)
print("4. INITIAL MODEL TESTING (UNTRAINED)")
print("=" * 60)

# Initialize untrained model
model = GPTModel(GPT_CONFIG_124M)
model.eval()
model.to(device)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Test text generation with untrained model
start_context = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["ctx_len"]
)
print("Untrained model output:")
print(token_ids_to_text(token_ids, tokenizer))

# ============================================================================
# 5. MODEL INFERENCE TESTING
# ============================================================================
print("\n" + "=" * 60)
print("5. MODEL INFERENCE TESTING")
print("=" * 60)

# Test inputs and targets
inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [588,  428,  11311]]) #  " really like chocolate"]

# Forward pass
with torch.no_grad():
    logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)
    predicted_token_ids = torch.argmax(probas, dim=-1, keepdim=True)

print("Logits shape:", logits.shape)
print("Token predictions:", predicted_token_ids)
print(f"Target batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Output batch 1: {token_ids_to_text(predicted_token_ids[0].flatten(), tokenizer)}")

# ============================================================================
# 6. DATA PREPARATION
# ============================================================================
print("\n" + "=" * 60)
print("6. DATA PREPARATION")
print("=" * 60)

# Load training data
with open("src/resources/the-verdict.txt", "r", encoding="utf-8") as f:
    text_data = f.read()

print(f"First 100 characters: {text_data[:99]}")
print(f"Last 100 characters: {text_data[-99:]}")

# Data statistics
total_char = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print(f"Characters: {total_char}")
print(f"Tokens: {total_tokens}")

# Split data into train/validation sets
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# ============================================================================
# 7. DATA LOADERS
# ============================================================================
print("\n" + "=" * 60)
print("7. DATA LOADERS")
print("=" * 60)

torch.manual_seed(123)

# Create data loaders
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

# Display data loader information
print("Train loader batches:")
for i, (x, y) in enumerate(train_loader):
    print(f"Batch {i}: {x.shape}, {y.shape}")
    if i >= 4:  # Show only first 5 batches
        break

print("\nValidation loader batches:")
for i, (x, y) in enumerate(val_loader):
    print(f"Batch {i}: {x.shape}, {y.shape}")

# Count total tokens
train_tokens = sum(input_batch.numel() for input_batch, _ in train_loader)
val_tokens = sum(input_batch.numel() for input_batch, _ in val_loader)

print(f"Training tokens: {train_tokens}")
print(f"Validation tokens: {val_tokens}")
print(f"Total tokens: {train_tokens + val_tokens}")

# ============================================================================
# 8. LOSS CALCULATION (UNTRAINED MODEL)
# ============================================================================
print("\n" + "=" * 60)
print("8. LOSS CALCULATION (UNTRAINED MODEL)")
print("=" * 60)

torch.manual_seed(123)
train_loss = calc_loss_loader(train_loader, model, device)
val_loss = calc_loss_loader(val_loader, model, device)

print(f"Training loss: {train_loss}")
print(f"Validation loss: {val_loss}")

# ============================================================================
# 9. TRAINING CODE (OPTIONAL)
# ============================================================================
print("\n" + "=" * 60)
print("9. TRAINING CODE")
print("=" * 60)

# Training the model from scratch
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=1,
    start_context="Every effort moves you", tokenizer=tokenizer
)
print("Training complete.")

# ============================================================================
# 10. GENERATION WITH TEMPERATURE SCALING AND TOP K (OPTIONAL)
# ============================================================================
print("\n" + "=" * 60)
print("10. GENERATION WITH TEMPERATURE SCALING AND TOP K")
print("=" * 60)

# Advanced generation with temperature scaling and top-k sampling
torch.manual_seed(123)

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["ctx_len"],
    top_k=25,
    temperature=1.4
)

print("Output text with temperature scaling and top-k:")
print(token_ids_to_text(token_ids, tokenizer))

# ============================================================================
# 11. DEPENDENCY VERSIONS
# ============================================================================
print("\n" + "=" * 60)
print("11. DEPENDENCY VERSIONS")
print("=" * 60)

try:
    print("TensorFlow version:", version("tensorflow"))
except Exception:
    print("TensorFlow version: Not installed")
print("tqdm version:", version("tqdm"))

# ============================================================================
# 12. LOADING PRE-TRAINED GPT-2 WEIGHTS
# ============================================================================
print("\n" + "=" * 60)
print("12. LOADING PRE-TRAINED GPT-2 WEIGHTS")
print("=" * 60)

# Load pre-trained weights and settings
settings, params, _ = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
print("Settings:", settings)
print("Parameters dictionary keys:", params.keys())
print("Token embedding shape:", params["wte"].shape)

# ============================================================================
# 13. CREATE AND LOAD PRE-TRAINED MODEL
# ============================================================================
print("\n" + "=" * 60)
print("13. CREATING PRE-TRAINED MODEL")
print("=" * 60)

# Configure model for pre-trained weights
model_name = "gpt2-small"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(MODEL_CONFIGS[model_name])
NEW_CONFIG.update({"ctx_len": 1024, "qkv_bias": True})

# Create and load pre-trained model
gpt = GPTModel(NEW_CONFIG)
gpt.eval()
load_weights_into_gpt(gpt, params)
gpt.to(device)

print(f"Model created with config: {model_name}")
print(f"Context length: {NEW_CONFIG['ctx_len']}")

# ============================================================================
# 14. TEXT GENERATION WITH PRE-TRAINED MODEL
# ============================================================================
print("\n" + "=" * 60)
print("14. TEXT GENERATION WITH PRE-TRAINED MODEL")
print("=" * 60)

torch.manual_seed(123)

# Generate text with advanced parameters
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=NEW_CONFIG["ctx_len"],
    top_k=50,
    temperature=1.5
)

print("Generated text:")
print(token_ids_to_text(token_ids, tokenizer))

print("\n" + "=" * 60)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("=" * 60)