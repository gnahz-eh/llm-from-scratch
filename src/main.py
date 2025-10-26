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
import time
import sys
import os

# Add src directory to path for proper imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

from src.model import GPTModel
from src.utils.data_loader import create_dataloader_v1
from src.utils.train import train_model_simple, calc_loss_loader, download_and_load_gpt2, load_weights_into_gpt
from src.utils.token import text_to_token_ids, token_ids_to_text
from src.utils.generate_text import generate_text_simple, generate

# Import UI components
try:
    from src.ui.web_app import (
        start_web_server, log_section_start, log_section_complete, 
        log_training_epoch, log_generation_result, log_message
    )
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False
    # Define dummy functions if UI is not available
    def start_web_server(*args, **kwargs): pass
    def log_section_start(*args, **kwargs): pass
    def log_section_complete(*args, **kwargs): pass
    def log_training_epoch(*args, **kwargs): pass
    def log_generation_result(*args, **kwargs): pass
    def log_message(*args, **kwargs): pass

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
# 3.5. WEB UI STARTUP
# ============================================================================
print("Starting web UI dashboard...")
log_message("🚀 Starting LLM Training Dashboard")
log_message(f"📱 Device: {device}")

# Start web server in background
if UI_AVAILABLE:
    try:
        server_thread = start_web_server(port=5000)
        log_message("🌐 Web dashboard available at http://127.0.0.1:5000", "success")
        time.sleep(2)  # Give server time to start
    except Exception as e:
        log_message(f"⚠️ Could not start web server: {e}", "warning")
        print(f"Warning: Web UI not available - {e}")
else:
    print("Web UI not available - Flask not installed")

# ============================================================================
# 4. INITIAL MODEL TESTING (UNTRAINED)
# ============================================================================
log_section_start(4, "Initial Model Testing (Untrained)")
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
log_section_complete(4, "Initial Model Testing (Untrained)")

# ============================================================================
# 5. MODEL INFERENCE TESTING
# ============================================================================
log_section_start(5, "Model Inference Testing")
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
log_section_complete(5, "Model Inference Testing")

# ============================================================================
# 6. DATA PREPARATION
# ============================================================================
log_section_start(6, "Data Preparation")
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
log_section_complete(6, "Data Preparation")

# ============================================================================
# 7. DATA LOADERS
# ============================================================================
log_section_start(7, "Data Loaders")
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
log_section_complete(7, "Data Loaders")

# ============================================================================
# 8. LOSS CALCULATION (UNTRAINED MODEL)
# ============================================================================
log_section_start(8, "Loss Calculation (Untrained Model)")
print("\n" + "=" * 60)
print("8. LOSS CALCULATION (UNTRAINED MODEL)")
print("=" * 60)

torch.manual_seed(123)
train_loss = calc_loss_loader(train_loader, model, device)
val_loss = calc_loss_loader(val_loader, model, device)

print(f"Training loss: {train_loss}")
print(f"Validation loss: {val_loss}")
log_section_complete(8, "Loss Calculation (Untrained Model)")

# ============================================================================
# 9. TRAINING CODE (OPTIONAL)
# ============================================================================
log_section_start(9, "Training Code")
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
log_section_complete(9, "Training Code")

# ============================================================================
# 10. GENERATION WITH TEMPERATURE SCALING AND TOP K (OPTIONAL)
# ============================================================================
log_section_start(10, "Generation with Temperature Scaling and Top K")
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
generated_text = token_ids_to_text(token_ids, tokenizer)
print(generated_text)

# Log generation result to UI
log_generation_result("Every effort moves you", generated_text, 1.4, 25)
log_section_complete(10, "Generation with Temperature Scaling and Top K")

# ============================================================================
# 11. DEPENDENCY VERSIONS
# ============================================================================
log_section_start(11, "Dependency Versions")
print("\n" + "=" * 60)
print("11. DEPENDENCY VERSIONS")
print("=" * 60)

try:
    print("TensorFlow version:", version("tensorflow"))
except Exception:
    print("TensorFlow version: Not installed")
print("tqdm version:", version("tqdm"))
log_section_complete(11, "Dependency Versions")

# ============================================================================
# 12. LOADING PRE-TRAINED GPT-2 WEIGHTS
# ============================================================================
log_section_start(12, "Loading Pre-trained GPT-2 Weights")
print("\n" + "=" * 60)
print("12. LOADING PRE-TRAINED GPT-2 WEIGHTS")
print("=" * 60)

# Load pre-trained weights and settings
settings, params, _ = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
print("Settings:", settings)
print("Parameters dictionary keys:", params.keys())
print("Token embedding shape:", params["wte"].shape)
log_section_complete(12, "Loading Pre-trained GPT-2 Weights")

# ============================================================================
# 13. CREATE AND LOAD PRE-TRAINED MODEL
# ============================================================================
log_section_start(13, "Create and Load Pre-trained Model")
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
log_section_complete(13, "Create and Load Pre-trained Model")

# ============================================================================
# 14. TEXT GENERATION WITH PRE-TRAINED MODEL
# ============================================================================
log_section_start(14, "Text Generation with Pre-trained Model")
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
final_generated_text = token_ids_to_text(token_ids, tokenizer)
print(final_generated_text)

# Log final generation result
log_generation_result("Every effort moves you", final_generated_text, 1.5, 50)
log_section_complete(14, "Text Generation with Pre-trained Model")

print("\n" + "=" * 60)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("=" * 60)
log_message("🎉 All sections completed successfully!", "success")

# Keep the web server running for continued access to results
if UI_AVAILABLE:
    log_message("🌐 Dashboard will remain active for viewing results", "info")
    log_message("📊 Refresh the page to view final results", "info")
    log_message("🛑 Press Ctrl+C in terminal to stop the dashboard", "info")
    print("\n" + "=" * 50)
    print("🌐 WEB DASHBOARD STILL RUNNING")
    print("📊 Visit: http://127.0.0.1:5000")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Keep the main thread alive to maintain the web server
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log_message("👋 Dashboard stopped by user", "info")
        print("\nDashboard stopped. Goodbye!")