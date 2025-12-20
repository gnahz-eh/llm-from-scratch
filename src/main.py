"""
Main script for GPT model training and text generation.
This script demonstrates the complete pipeline from data preparation to text generation.
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================

# Standard library imports
import os
import sys
import time
from importlib.metadata import version

# Third-party imports
import torch
import tiktoken

# Add src directory to path for proper imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# Local imports
from src.model import GPTModel
from src.utils.data_loader import create_dataloader_v1
from src.utils.train import (
    train_model_simple, 
    calc_loss_loader, 
    download_and_load_gpt2, 
    load_weights_into_gpt
)
from src.utils.token import text_to_token_ids, token_ids_to_text
from src.utils.generate_text import generate_text_simple, generate

# Import UI components with fallback
try:
    from src.ui.web_app import (
        start_web_server, 
        log_section_start, 
        log_section_complete, 
        log_training_epoch, 
        log_generation_result, 
        log_message,
        log_model_result,
        log_inference_test,
        log_data_stats,
        log_loss_stats,
        log_pretrained_loading
    )
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False
    
    # Define dummy functions if UI is not available
    def start_web_server(*args, **kwargs): 
        pass
    
    def log_section_start(*args, **kwargs): 
        pass
    
    def log_section_complete(*args, **kwargs): 
        pass
    
    def log_training_epoch(*args, **kwargs): 
        pass
    
    def log_generation_result(*args, **kwargs): 
        pass
    
    def log_message(*args, **kwargs): 
        pass
    
    def log_model_result(*args, **kwargs): 
        pass
    
    def log_inference_test(*args, **kwargs): 
        pass
    
    def log_data_stats(*args, **kwargs): 
        pass
    
    def log_loss_stats(*args, **kwargs): 
        pass
    
    def log_pretrained_loading(*args, **kwargs): 
        pass

# ============================================================================
# 2. GLOBAL VARIABLES AND CONFIGURATION
# ============================================================================

# Base GPT configuration (124M parameter model)
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "ctx_len": 256,        # Context length (shortened from 1024 for demo)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

# Model variants configuration
MODEL_CONFIGS = {
    "gpt2-small":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl":     {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Global variables - Device and hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables - Data processing
tokenizer = None
text_data = None
train_data = None
val_data = None
train_loader = None
val_loader = None

# Global variables - Models
model = None # self-trained model
gpt = None # pre-trained model from GPT-2
NEW_CONFIG = None

# Global variables - UI
server_thread = None

# Global variables - Test prompts
TEST_PROMPT = "Every effort moves you"

# ============================================================================
# 3. FUNCTION DEFINITIONS
# ============================================================================

def setup_device_and_seeding():
    """Setup device and seed for reproducibility."""
    log_section_start(1, "Device Setup & Initialization")
    torch.manual_seed(123)  # For reproducibility
    log_message(f"📱 Device: {device}")
    log_section_complete(1, "Device Setup & Initialization")


def start_web_ui():
    """Start the web UI dashboard."""
    global server_thread
    
    print("Starting web UI dashboard...")
    log_message("🚀 Starting LLM Training Dashboard")
    
    # Start web server in background
    if UI_AVAILABLE:
        try:
            server_thread = start_web_server(port=5000)
            log_message(
                "🌐 Web dashboard available at http://127.0.0.1:5000", 
                "success"
            )
            time.sleep(2)  # Give server time to start
        except Exception as e:
            log_message(f"⚠️ Could not start web server: {e}", "warning")
            print(f"Warning: Web UI not available - {e}")
    else:
        print("Web UI not available - Flask not installed")


def initialize_tokenizer():
    """Initialize the GPT-2 tokenizer."""
    global tokenizer
    log_section_start(2, "Tokenizer Initialization")
    tokenizer = tiktoken.get_encoding("gpt2")
    log_message("🔤 GPT-2 tokenizer initialized", "success")
    log_section_complete(2, "Tokenizer Initialization")

def test_initial_model():
    """Test initial untrained model generation capabilities."""
    global model
    
    log_section_start(3, "Initial Model Testing (Untrained)")
    print("=" * 60)
    print("3. INITIAL MODEL TESTING (UNTRAINED)")
    print("=" * 60)

    # Initialize untrained model
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    model.to(device)

    # Test text generation with untrained model
    start_context = TEST_PROMPT
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["ctx_len"]
    )
    
    untrained_output = token_ids_to_text(token_ids, tokenizer)
    print("Untrained model output:")
    print(untrained_output)
    
    # Log untrained result to UI for later comparison
    log_model_result('untrained', untrained_output)
    log_message(f"🤖 UNTRAINED OUTPUT: '{untrained_output}'", "warning")
    
    log_section_complete(3, "Initial Model Testing (Untrained)")


def test_model_inference():
    """Test model inference with specific input sequences."""
    log_section_start(4, "Model Inference Testing")
    print("\n" + "=" * 60)
    print("4. MODEL INFERENCE TESTING")
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
    
    target_text = token_ids_to_text(targets[0], tokenizer)
    output_text = token_ids_to_text(predicted_token_ids[0].flatten(), tokenizer)
    
    print(f"Target batch 1: {target_text}")
    print(f"Output batch 1: {output_text}")
    
    # Calculate accuracy
    correct_tokens = (predicted_token_ids[0].flatten() == targets[0]).sum().item()
    total_tokens = targets[0].numel()
    accuracy = correct_tokens / total_tokens * 100
    
    print(f"Token-level accuracy: {accuracy:.1f}% ({correct_tokens}/{total_tokens})")
    
    # Log inference test results to UI
    log_inference_test(target_text, output_text, accuracy)
    log_message(f"🎯 TARGET: '{target_text}'", "info")
    log_message(f"🤖 UNTRAINED PREDICTION: '{output_text}'", "warning")
    log_message(f"📊 Token accuracy: {accuracy:.1f}%", "info")

    log_section_complete(4, "Model Inference Testing")

def prepare_data():
    """Load and prepare training data."""
    global text_data, train_data, val_data
    
    log_section_start(5, "Data Preparation")
    print("\n" + "=" * 60)
    print("5. DATA PREPARATION")
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
    
    # Log data stats to UI (will be completed in create_data_loaders)
    log_message(f"📊 Text data loaded: {total_char:,} characters, {total_tokens:,} tokens")

    # Split data into train/validation sets
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    log_message(f"🔄 Data split: {train_ratio*100:.0f}% training, {(1-train_ratio)*100:.0f}% validation")
    log_section_complete(5, "Data Preparation")

def create_data_loaders():
    """Create training and validation data loaders."""
    global train_loader, val_loader
    
    log_section_start(6, "Data Loaders")
    print("\n" + "=" * 60)
    print("6. DATA LOADERS")
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
    total_chars = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))

    print(f"Training tokens: {train_tokens}")
    print(f"Validation tokens: {val_tokens}")
    print(f"Total tokens: {train_tokens + val_tokens}")
    
    # Log data statistics to UI
    log_data_stats(total_chars, total_tokens, train_tokens, val_tokens)
    log_message(f"📊 Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    log_section_complete(6, "Data Loaders")

def calculate_initial_loss():
    """Calculate loss for untrained model."""
    log_section_start(7, "Loss Calculation (Untrained Model)")
    print("\n" + "=" * 60)
    print("7. LOSS CALCULATION (UNTRAINED MODEL)")
    print("=" * 60)

    torch.manual_seed(123)
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

    print(f"Training loss: {train_loss}")
    print(f"Validation loss: {val_loss}")
    
    # Log initial loss statistics to UI
    log_loss_stats('initial', train_loss, val_loss)
    
    log_section_complete(7, "Loss Calculation (Untrained Model)")

def train_model():
    """Train the model from scratch."""
    log_section_start(8, "Training Code")
    print("\n" + "=" * 60)
    print("8. TRAINING CODE")
    print("=" * 60)

    # Training the model from scratch
    torch.manual_seed(123)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

    num_epochs = 1
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=1,
        start_context=TEST_PROMPT, tokenizer=tokenizer
    )

    print("Training complete.")
    log_section_complete(8, "Training Code")

def test_temperature_generation():
    """Test generation with temperature scaling and top-k sampling."""
    log_section_start(9, "Generation with Temperature Scaling and Top K")
    print("\n" + "=" * 60)
    print("9. GENERATION WITH TEMPERATURE SCALING AND TOP K")
    print("=" * 60)

    # Advanced generation with temperature scaling and top-k sampling
    torch.manual_seed(123)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(TEST_PROMPT, tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["ctx_len"],
        top_k=25,
        temperature=1.4
    )

    print("Output text with temperature scaling and top-k:")
    generated_text = token_ids_to_text(token_ids, tokenizer)
    print(generated_text)
    
    # Log advanced generation result to UI for comparison
    log_model_result('advanced', generated_text)
    log_generation_result(TEST_PROMPT, generated_text, 1.4, 25)
    
    log_section_complete(9, "Generation with Temperature Scaling and Top K")

def check_dependency_versions():
    """Check and display dependency versions."""
    log_section_start(10, "Dependency Versions")
    print("\n" + "=" * 60)
    print("10. DEPENDENCY VERSIONS")
    print("=" * 60)

    try:
        print("TensorFlow version:", version("tensorflow"))
    except Exception:
        print("TensorFlow version: Not installed")
    print("tqdm version:", version("tqdm"))
    log_section_complete(10, "Dependency Versions")

def load_pretrained_weights():
    """Load pre-trained GPT-2 weights."""
    log_section_start(11, "Loading Pre-trained GPT-2 Weights")
    print("\n" + "=" * 60)
    print("11. LOADING PRE-TRAINED GPT-2 WEIGHTS")
    print("=" * 60)

    # Log loading progress
    log_pretrained_loading("Starting GPT-2 124M model download...")
    
    # Load pre-trained weights and settings
    settings, params, _ = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    
    log_pretrained_loading("GPT-2 weights loaded successfully!")
    
    print("Settings:", settings)
    print("Parameters dictionary keys:", params.keys())
    print("Token embedding shape:", params["wte"].shape)
    
    log_pretrained_loading(f"Loaded {len(params)} parameter tensors")
    log_message(f"✅ Pre-trained weights loaded: {len(params)} tensors", "success")
    
    log_section_complete(11, "Loading Pre-trained GPT-2 Weights")
    
    return settings, params

def create_pretrained_model(params):
    """Create and load pre-trained model."""
    global gpt, NEW_CONFIG
    
    log_section_start(12, "Create and Load Pre-trained Model")
    print("\n" + "=" * 60)
    print("12. CREATING PRE-TRAINED MODEL")
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
    log_section_complete(12, "Create and Load Pre-trained Model")

def test_pretrained_generation():
    """Test text generation with pre-trained model."""
    log_section_start(13, "Text Generation with Pre-trained Model")
    print("\n" + "=" * 60)
    print("13. TEXT GENERATION WITH PRE-TRAINED MODEL")
    print("=" * 60)

    torch.manual_seed(123)

    # Generate text with advanced parameters
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(TEST_PROMPT, tokenizer),
        max_new_tokens=25,
        context_size=NEW_CONFIG["ctx_len"],
        top_k=50,
        temperature=1.5
    )

    print("Generated text:")
    final_generated_text = token_ids_to_text(token_ids, tokenizer)
    print(final_generated_text)
    
    # Log pretrained generation result for comparison
    log_model_result('pretrained', final_generated_text)
    log_generation_result(TEST_PROMPT, final_generated_text, 1.5, 50)
    
    # Log comparison summary
    log_message("🔄 COMPARISON SUMMARY:", "info")
    log_message("  • Untrained model results captured", "warning")
    log_message("  • Trained model results captured", "info")
    log_message("  • Advanced generation results captured", "info")
    log_message("  • Pre-trained model results captured", "success")
    
    log_section_complete(13, "Text Generation with Pre-trained Model")
    
    return final_generated_text

def keep_web_server_running():
    """Keep the web server running for continued access to results."""
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

# ============================================================================
# 4. MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function that orchestrates the entire GPT pipeline.
    
    This function coordinates the complete workflow from initialization
    to final text generation, including:
    - Device setup and initialization
    - Model testing and data preparation  
    - Training from scratch
    - Pre-trained model evaluation
    - Web dashboard management
    """
    try:
        # ================================================================
        # SECTION 1: SETUP AND INITIALIZATION
        # ================================================================
        setup_device_and_seeding()
        start_web_ui()
        initialize_tokenizer()
        
        # ================================================================
        # SECTION 2: MODEL TESTING AND DATA PREPARATION
        # ================================================================
        test_initial_model() # Try to generate text with untrained model
        test_model_inference() # Try to generate one token with untrained model
        prepare_data() # Read the local test data and split into train/val
        create_data_loaders() # Create PyTorch DataLoader for train/val sets
        calculate_initial_loss() # Calculate loss with untrained model
        
        # ================================================================
        # SECTION 3: TRAINING (OPTIONAL)
        # ================================================================
        train_model() # Train the model from scratch, and print related results
        test_temperature_generation() # Test generation with temperature scaling and top-k sampling
        
        # ================================================================
        # SECTION 4: PRE-TRAINED MODEL TESTING
        # ================================================================
        check_dependency_versions() # Check versions of key dependencies
        settings, params = load_pretrained_weights() # Load pre-trained GPT-2 weights
        create_pretrained_model(params) # Create model and load pre-trained weights
        final_generated_text = test_pretrained_generation() # Test generation with pre-trained model
        
        # ================================================================
        # SECTION 5: COMPLETION
        # ================================================================
        print("\n" + "=" * 60)
        print("SCRIPT COMPLETED SUCCESSFULLY")
        print("=" * 60)
        log_message("🎉 All sections completed successfully!", "success")
        
        # ================================================================
        # SECTION 6: KEEP WEB SERVER RUNNING
        # ================================================================
        keep_web_server_running()
        
    except KeyboardInterrupt:
        log_message("👋 Script interrupted by user", "warning")
        print("\nScript stopped by user. Goodbye!")
    except Exception as e:
        log_message(f"❌ Error occurred: {e}", "error")
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()