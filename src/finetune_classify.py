"""
Fine-tuning GPT model for spam classification.

This script demonstrates the complete pipeline for fine-tuning a pre-trained GPT model
for binary classification tasks (spam vs ham SMS messages).
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================

# Standard library imports
import os
import urllib.request
import zipfile
from pathlib import Path

# Third-party imports
import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader

# Local imports
from src.finetune import (
    download_and_unzip_spam_data,
    create_balanced_dataset,
    random_split,
    SpamDataset,
)
from src.model import GPTModel
from src.utils.generate_text import generate_text_simple, classify_review
from src.utils.token import text_to_token_ids, token_ids_to_text
from src.utils.train import (
    download_and_load_gpt2,
    load_weights_into_gpt,
    calc_accuracy_loader,
    calc_loss_loader_4_classification,
    train_classifier_simple
)


# ============================================================================
# 2. CONFIGURATION
# ============================================================================

# Dataset configuration
DATA_FILE_PATH = "./src/resources/sms_spam_collection/SMSSpamCollection.tsv"
DATA_OUTPUT_PATH = "./src/resources/sms_spam_collection/processed"
MODEL_OUTPUT_PATH = "./src/resources/self_trained_models"

# Model configuration
BASE_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "ctx_len": 1024,      # Context length
    "drop_rate": 0.0,     # Dropout rate
    "qkv_bias": True,     # Query-key-value bias
}

MODEL_CONFIGS = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Training configuration
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BATCH_SIZE = 8
NUM_WORKERS = 0
NUM_CLASSES = 2

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
# Test ratio is automatically calculated as 1 - train_ratio - val_ratio

# ============================================================================
# 3. FUNCTION DEFINITIONS
# ============================================================================

def load_and_prepare_data():
    """Load and prepare the SMS spam dataset."""
    print("Loading SMS spam dataset...")
    df = pd.read_csv(DATA_FILE_PATH, sep="\t", header=None, names=["Label", "Text"])
    print(f"Original dataset shape: {df.shape}")
    print(df.head())
    
    # Create balanced dataset
    balanced_df = create_balanced_dataset(df)
    print("\nBalanced dataset label distribution:")
    print(balanced_df["Label"].value_counts())
    
    # Convert labels to numeric values
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    
    return balanced_df


def split_and_save_data(balanced_df):
    """Split data into train/validation/test sets and save to CSV files."""
    print(f"\nSplitting data with ratios - Train: {TRAIN_RATIO}, Val: {VAL_RATIO}")
    
    train_df, validation_df, test_df = random_split(
        balanced_df, TRAIN_RATIO, VAL_RATIO
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(DATA_OUTPUT_PATH, exist_ok=True)
    
    # Save datasets
    train_df.to_csv(f"{DATA_OUTPUT_PATH}/train.csv", index=None)
    validation_df.to_csv(f"{DATA_OUTPUT_PATH}/validation.csv", index=None)
    test_df.to_csv(f"{DATA_OUTPUT_PATH}/test.csv", index=None)
    
    print(f"Datasets saved to {DATA_OUTPUT_PATH}/")
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(validation_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, validation_df, test_df


def initialize_tokenizer():
    """Initialize the GPT-2 tokenizer."""
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Test tokenizer with special token
    special_tokens = tokenizer.encode(
        "<|endoftext|>", 
        allowed_special={"<|endoftext|>"}
    )
    print(f"\nTokenizer initialized. Special token encoding: {special_tokens}")
    
    return tokenizer


def create_datasets_and_loaders(tokenizer):
    """Create datasets and data loaders for training, validation, and testing."""
    print(f"\nCreating datasets with batch size: {BATCH_SIZE}")
    
    # Create datasets
    train_dataset = SpamDataset(
        csv_file=f"{DATA_OUTPUT_PATH}/train.csv", 
        max_length=None, 
        tokenizer=tokenizer
    )
    
    print(f"Maximum sequence length: {train_dataset.max_length}")
    
    val_dataset = SpamDataset(
        csv_file=f"{DATA_OUTPUT_PATH}/validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )
    
    test_dataset = SpamDataset(
        csv_file=f"{DATA_OUTPUT_PATH}/test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )
    
    # Set random seed for reproducibility
    torch.manual_seed(123)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    
    # Display loader information
    print(f"Data loaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test batch dimensions
    for input_batch, target_batch in train_loader:
        print(f"  Input batch dimensions: {input_batch.shape}")
        print(f"  Label batch dimensions: {target_batch.shape}")
        break
    
    return train_loader, val_loader, test_loader, train_dataset


def load_pretrained_model():
    """Load and configure the pre-trained GPT model."""
    print(f"\nLoading pre-trained model: {CHOOSE_MODEL}")
    
    # Update base config with selected model
    config = BASE_CONFIG.copy()
    config.update(MODEL_CONFIGS[CHOOSE_MODEL])
    
    # Extract model size and download weights
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params, _ = download_and_load_gpt2(
        model_size=model_size, 
        models_dir="gpt2"
    )
    
    # Create and load model
    model = GPTModel(config)
    load_weights_into_gpt(model, params)
    model.eval()
    
    print(f"Model loaded successfully with config: {config}")
    
    return model, config


def test_text_generation(model, tokenizer, config):
    """Test text generation capabilities of the model."""
    print(f"\n{'='*60}")
    print("TESTING TEXT GENERATION")
    print(f"{'='*60}")
    
    # Test 1: Simple text generation
    print(f"\nTest 1 - Input: '{INPUT_PROMPT}'")
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(INPUT_PROMPT, tokenizer),
        max_new_tokens=15,
        context_size=config["ctx_len"]
    )
    print("Generated text:")
    print(token_ids_to_text(token_ids, tokenizer))
    
    # Test 2: Spam classification prompt
    print(f"\nTest 2 - Spam Classification Test:")
    spam_prompt = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
        " Answer with 'yes' or 'no'."
    )
    
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(spam_prompt, tokenizer),
        max_new_tokens=23,
        context_size=config["ctx_len"]
    )
    print("Generated response:")
    print(token_ids_to_text(token_ids, tokenizer))


def prepare_model_for_finetuning(model, config):
    """Prepare the model for fine-tuning by freezing parameters and adding classification head."""
    print(f"\n{'='*60}")
    print("PREPARING MODEL FOR FINE-TUNING")
    print(f"{'='*60}")
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Add classification head
    model.out_head = torch.nn.Linear(
        in_features=config["emb_dim"], 
        out_features=NUM_CLASSES
    )
    
    # Unfreeze last transformer block
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    
    # Unfreeze final normalization layer
    for param in model.final_norm.parameters():
        param.requires_grad = True
    
    print(f"Model prepared for fine-tuning:")
    print(f"  Added classification head: {config['emb_dim']} -> {NUM_CLASSES}")
    print(f"  Unfrozen last transformer block and final normalization")
    
    return model


def test_model_output(model, tokenizer):
    """Test the model output with classification head."""
    print(f"\n{'='*60}")
    print("TESTING MODEL OUTPUT")
    print(f"{'='*60}")
    
    # Set random seed for reproducibility
    torch.manual_seed(123)
    
    # Test input
    test_text = "Do you have time"
    inputs = tokenizer.encode(test_text)
    inputs = torch.tensor(inputs).unsqueeze(0)
    
    print(f"Input text: '{test_text}'")
    print(f"Input tokens: {inputs}")
    print(f"Input dimensions: {inputs.shape}")  # (batch_size, num_tokens)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)
    
    print(f"Output shape: {outputs.shape}")  # (batch_size, num_tokens, num_classes)
    print("Model outputs:")
    print(outputs)


def evaluate_model_before_training(model, train_loader, val_loader, test_loader, device):
    """Evaluate model accuracy and loss before fine-tuning."""
    print(f"\n{'='*60}")
    print("EVALUATING MODEL BEFORE TRAINING")
    print(f"{'='*60}")
    
    # Set random seed for reproducibility
    torch.manual_seed(123)
    
    # Calculate accuracies
    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    
    # Calculate losses
    with torch.no_grad():
        train_loss = calc_loss_loader_4_classification(
            train_loader, model, device, num_batches=5
        )
        val_loss = calc_loss_loader_4_classification(
            val_loader, model, device, num_batches=5
        )
        test_loss = calc_loss_loader_4_classification(
            test_loader, model, device, num_batches=5
        )

    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")
    
    return {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss
    }


def train_classification_model(model, train_loader, val_loader, device, tokenizer):
    """Train the classification model with fine-tuning."""
    print(f"\n{'='*60}")
    print("TRAINING CLASSIFICATION MODEL")
    print(f"{'='*60}")
    
    import time
    
    start_time = time.time()
    
    # Set random seed for reproducibility
    torch.manual_seed(123)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    
    # Training configuration
    num_epochs = 5
    
    print(f"Training configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: 5e-5")
    print(f"  Weight decay: 0.1")
    print(f"  Optimizer: AdamW")
    
    # Train the model
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
        tokenizer=tokenizer
    )
    
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'examples_seen': examples_seen,
        'execution_time': execution_time_minutes
    }


def evaluate_model_after_training(model, train_loader, val_loader, test_loader, device):
    """Evaluate model performance after fine-tuning."""
    print(f"\n{'='*60}")
    print("EVALUATING MODEL AFTER TRAINING")
    print(f"{'='*60}")
    
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Final Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Final Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Final Test accuracy: {test_accuracy*100:.2f}%")
    
    return {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy
    }


def test_classification_examples(model, tokenizer, device, max_length):
    """Test the trained model with example texts."""
    print(f"\n{'='*60}")
    print("TESTING CLASSIFICATION EXAMPLES")
    print(f"{'='*60}")
    
    # Test spam example
    spam_text = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )
    
    print(f"\nTesting spam example:")
    print(f"Text: '{spam_text}'")
    spam_result = classify_review(
        spam_text, model, tokenizer, device, max_length=max_length
    )
    print(f"Classification result: {spam_result}")
    
    # Test ham example
    ham_text = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )
    
    print(f"\nTesting ham example:")
    print(f"Text: '{ham_text}'")
    ham_result = classify_review(
        ham_text, model, tokenizer, device, max_length=max_length
    )
    print(f"Classification result: {ham_result}")
    
    return {'spam_example': spam_result, 'ham_example': ham_result}


def save_trained_model(model):
    """Save the trained model to disk."""
    print(f"\n{'='*60}")
    print("SAVING TRAINED MODEL")
    print(f"{'='*60}")
    
    # Create output directory if it doesn't exist
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    
    model_path = f"{MODEL_OUTPUT_PATH}/review_classifier.pth"
    torch.save(model.state_dict(), model_path)
    
    print(f"Model saved successfully to: {model_path}")
    
    return model_path


# ============================================================================
# 4. MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function that orchestrates the spam classification fine-tuning pipeline.
    
    This function coordinates the complete workflow:
    - Data loading and preprocessing
    - Dataset creation and splitting
    - Model loading and configuration
    - Fine-tuning preparation
    - Testing and validation
    """
    try:
        print("=" * 80)
        print("SPAM CLASSIFICATION FINE-TUNING PIPELINE")
        print("=" * 80)
        
        # ================================================================
        # SECTION 1: DATA PREPARATION
        # ================================================================
        balanced_df = load_and_prepare_data()
        train_df, validation_df, test_df = split_and_save_data(balanced_df)
        
        # ================================================================
        # SECTION 2: TOKENIZER AND DATASET CREATION
        # ================================================================
        tokenizer = initialize_tokenizer()
        train_loader, val_loader, test_loader, train_dataset = create_datasets_and_loaders(tokenizer)
        
        # ================================================================
        # SECTION 3: MODEL LOADING AND TESTING
        # ================================================================
        model, config = load_pretrained_model()
        test_text_generation(model, tokenizer, config)
        
        # ================================================================
        # SECTION 4: FINE-TUNING PREPARATION
        # ================================================================
        model = prepare_model_for_finetuning(model, config)
        test_model_output(model, tokenizer)
        
        # ================================================================
        # SECTION 5: MODEL TRAINING AND EVALUATION
        # ================================================================
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Evaluate model before training
        pre_training_metrics = evaluate_model_before_training(
            model, train_loader, val_loader, test_loader, device
        )
        
        # Train the classification model
        training_results = train_classification_model(
            model, train_loader, val_loader, device, tokenizer
        )
        
        # Evaluate model after training
        post_training_metrics = evaluate_model_after_training(
            model, train_loader, val_loader, test_loader, device
        )
        
        # Test with example classifications
        classification_examples = test_classification_examples(
            model, tokenizer, device, train_dataset.max_length
        )
        
        # Save the trained model
        model_path = save_trained_model(model)
        
        # ================================================================
        # SECTION 6: COMPLETION
        # ================================================================
        print(f"\n{'='*80}")
        print("FINE-TUNING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        # Print summary of results
        print("\n📊 TRAINING SUMMARY:")
        print(f"  Training time: {training_results['execution_time']:.2f} minutes")
        print(f"  Model saved to: {model_path}")
        
        print("\n📈 PERFORMANCE IMPROVEMENT:")
        print(f"  Train accuracy: {pre_training_metrics['train_accuracy']*100:.2f}% → {post_training_metrics['train_accuracy']*100:.2f}%")
        print(f"  Validation accuracy: {pre_training_metrics['val_accuracy']*100:.2f}% → {post_training_metrics['val_accuracy']*100:.2f}%")
        print(f"  Test accuracy: {pre_training_metrics['test_accuracy']*100:.2f}% → {post_training_metrics['test_accuracy']*100:.2f}%")
        
        print("\n🎯 CLASSIFICATION EXAMPLES:")
        print(f"  Spam detection: {classification_examples['spam_example']}")
        print(f"  Ham detection: {classification_examples['ham_example']}")

    except Exception as e:
        print(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()