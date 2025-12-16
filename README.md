# 🤖 Nano-LLM

A complete implementation of a GPT-style language model built from scratch using PyTorch, featuring a modern web-based training dashboard for real-time monitoring and visualization.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey.svg)](https://flask.palletsprojects.com/)

> **Reference**: Based on the excellent work from [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka

## 🌟 Features

- **🏗️ Complete GPT Architecture**: Full implementation of transformer-based language model
- **🎯 Training Pipeline**: End-to-end training process with data loading, optimization, and evaluation
- **📊 Real-time Dashboard**: Beautiful web interface for monitoring training progress
- **🔧 Modular Design**: Clean, modular codebase with separate components for attention, transformers, and utilities
- **⚡ Pre-trained Models**: Support for loading and using pre-trained GPT-2 weights
- **🎨 Text Generation**: Multiple text generation strategies including temperature scaling and top-k sampling
- **📈 Visualization**: Live training loss charts and epoch-by-epoch progress tracking

## 🏗️ Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Nano-LLM Training System                    │
├─────────────────────┬───────────────────┬───────────────────┤
│   Web Dashboard     │   Core ML Model   │   Training Utils  │
│                     │                   │                   │
│ ┌─────────────────┐ │ ┌───────────────┐ │ ┌───────────────┐ │
│ │ Flask Server    │ │ │ GPT Model     │ │ │ Data Loaders  │ │
│ │ Real-time UI    │ │ │ Transformers  │ │ │ Tokenization  │ │
│ │ Progress Track  │ │ │ Attention     │ │ │ Training Loop │ │
│ └─────────────────┘ │ └───────────────┘ │ └───────────────┘ │
└─────────────────────┴───────────────────┴───────────────────┘
```

### Model Architecture

```
GPTModel
├── Token Embedding (vocab_size → emb_dim)
├── Position Embedding (ctx_len → emb_dim)
├── Dropout Layer
├── Transformer Blocks (12 layers)
│   ├── Multi-Head Attention
│   │   ├── Self Attention
│   │   ├── Causal Attention
│   │   └── Query/Key/Value Projections
│   ├── Feed Forward Network
│   └── Layer Normalization
├── Final Layer Norm
└── Output Head (emb_dim → vocab_size)
```

## 📁 Project Structure

```
nano-llm/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 LICENSE                      # MIT license
├── 🔧 .gitignore                   # Git ignore rules
├── 📁 gpt2/                        # Pre-trained GPT-2 weights
│   └── 124M/                       # 124M parameter model files
├── 📁 src/                         # Source code
│   ├── 📄 main.py                  # Main training script
│   ├── 📁 model/                   # Model architecture
│   │   ├── 📄 __init__.py
│   │   └── 📄 gpt_model.py         # GPT model implementation
│   ├── 📁 attention/               # Attention mechanisms
│   │   ├── 📄 __init__.py
│   │   ├── 📄 self_attention.py    # Self-attention implementation
│   │   ├── 📄 causal_attention.py  # Causal attention
│   │   └── 📄 multi_head_attention.py # Multi-head attention
│   ├── 📁 transformer/             # Transformer components
│   │   ├── 📄 __init__.py
│   │   ├── 📄 transformer_block.py # Transformer block
│   │   ├── 📄 feed_forward.py      # Feed-forward network
│   │   └── 📄 layer_norm.py        # Layer normalization
│   ├── 📁 tokenizer/               # Tokenization utilities
│   │   ├── 📄 __init__.py
│   │   └── 📄 simple_tokenizer.py  # Simple tokenizer
│   ├── 📁 utils/                   # Training utilities
│   │   ├── 📄 __init__.py
│   │   ├── 📄 data_loader.py       # Data loading functions
│   │   ├── 📄 train.py             # Training loop and utilities
│   │   ├── 📄 generate_text.py     # Text generation functions
│   │   └── 📄 token.py             # Token processing utilities
│   ├── 📁 ui/                      # Web dashboard
│   │   ├── 📄 __init__.py
│   │   ├── 📄 web_app.py           # Flask web application
│   │   ├── 📄 README.md            # Dashboard documentation
│   │   ├── 📄 IMPROVEMENTS.md      # Recent improvements
│   │   └── 📁 templates/
│   │       └── 📄 dashboard.html   # Web interface template
│   ├── 📁 resources/               # Training data and resources
│   │   ├── 📄 the-verdict.txt      # Sample training text
│   │   └── 📄 training-result.md   # Training results documentation
│   └── 📁 book/                    # Jupyter notebooks for learning
│       ├── 📁 ch02/                # Chapter 2 notebooks
│       ├── 📁 ch03/                # Chapter 3 notebooks
│       ├── 📁 ch04/                # Chapter 4 notebooks
│       └── 📁 ch05/                # Chapter 5 notebooks
└── 📁 .venv/                       # Python virtual environment
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- PyTorch 2.3+
- 4GB+ RAM (8GB+ recommended for training)
- CUDA-compatible GPU (optional, but recommended for faster training)

### 2. Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/gnahz-eh/nano-llm.git
   cd nano-llm
   ```

2. **Create and activate virtual environment** (recommended):
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Run Training with Dashboard

Execute the complete training pipeline with real-time web dashboard:

```bash
python -m src.main
```

This will:
- 🚀 Start the Flask web server on http://127.0.0.1:5000
- 🌐 Automatically open your browser to the dashboard
- 📊 Show real-time progress through 14 training sections
- 🎯 Train the model for 10 epochs
- 🎨 Generate text with both trained and pre-trained models
- 📈 Display live training loss charts
- 📝 Provide detailed logs and generation results

### 4. Monitor Progress

The web dashboard provides:
- **Section Progress**: Visual progress through all training stages
- **Training Statistics**: Real-time epoch, loss, and token counts
- **Loss Visualization**: Live updating training and validation loss charts
- **Text Generation**: Display of generated text with parameters
- **Live Logs**: Color-coded log stream with timestamps
- **Epoch Details**: Historical view of training epochs

## 🎯 Usage Examples

### Basic Training

```python
from src.model import GPTModel
from src.utils.train import train_model_simple
from src.utils.data_loader import create_dataloader_v1

# Configure model
config = {
    "vocab_size": 50257,
    "ctx_len": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# Initialize model
model = GPTModel(config)

# Create data loaders
train_loader = create_dataloader_v1(train_data, batch_size=2, max_length=256)
val_loader = create_dataloader_v1(val_data, batch_size=2, max_length=256)

# Train model
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=10, eval_freq=5, eval_iter=1
)
```

### Text Generation

```python
from src.utils.generate_text import generate
from src.utils.token import text_to_token_ids, token_ids_to_text
import tiktoken

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Generate text with temperature and top-k sampling
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=256,
    top_k=50,
    temperature=1.5
)

generated_text = token_ids_to_text(token_ids, tokenizer)
print(f"Generated: {generated_text}")
```

### Using Pre-trained GPT-2

```python
from src.utils.train import download_and_load_gpt2, load_weights_into_gpt

# Load pre-trained GPT-2 weights
settings, params, _ = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# Create model with pre-trained config
config = {
    "vocab_size": 50257,
    "ctx_len": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True
}

gpt = GPTModel(config)
load_weights_into_gpt(gpt, params)
gpt.eval()
```

## 📊 Web Dashboard Features

### Real-time Training Monitor

The included web dashboard provides a professional interface for monitoring your LLM training:

![Dashboard Features](https://img.shields.io/badge/Dashboard-Live%20Monitoring-brightgreen)

**Key Features:**
- 🎯 **Section Tracking**: Visual progress through 14 training sections
- 📈 **Live Charts**: Real-time training and validation loss visualization
- 🔢 **Statistics**: Current epoch, tokens processed, latest loss values
- 🎨 **Text Generation**: Display of model outputs with generation parameters
- 📝 **Activity Logs**: Color-coded live log stream
- 📊 **Epoch History**: Detailed epoch-by-epoch training progress
- 🎪 **Generation History**: Track multiple text generation results

**Technical Details:**
- Built with Flask and Chart.js
- Real-time updates every second via REST API
- Thread-safe progress tracking
- Responsive design for desktop and mobile
- Professional, clean interface
- Persistent access after training completion

### Dashboard Sections

1. **Section Progress**: Shows completion status of all training stages
2. **Training Statistics**: Live metrics and progress indicators  
3. **Training Loss Chart**: Interactive line charts with hover details
4. **Text Generation Results**: Input/output text with generation parameters
5. **Live Logs**: Real-time activity feed with timestamps

## 🧠 Model Details

### GPT Architecture Components

**Token and Position Embeddings**:
- Vocabulary size: 50,257 tokens (GPT-2 tokenizer)
- Context length: 256-1024 tokens
- Embedding dimension: 768

**Transformer Blocks** (12 layers):
- **Multi-Head Attention**: 12 attention heads
- **Causal Masking**: Prevents looking at future tokens
- **Feed Forward**: 4x expansion ratio (768 → 3072 → 768)
- **Layer Normalization**: Pre-normalization design
- **Residual Connections**: Skip connections for training stability

**Output Layer**:
- Linear projection from embedding to vocabulary space
- No bias in output layer (following GPT design)

### Training Configuration

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # GPT-2 tokenizer vocabulary
    "ctx_len": 256,         # Context length (can be 1024)
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of transformer layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Bias in attention projections
}
```

## 📈 Training Process

The training pipeline consists of 14 main sections:

1. **Library Dependencies Check**: Verify PyTorch and dependencies
2. **Model Configuration**: Set up model parameters and config
3. **Device Setup**: Configure GPU/CPU and random seeds
4. **Initial Model Testing**: Test untrained model behavior
5. **Model Inference Testing**: Forward pass validation
6. **Data Preparation**: Load and preprocess training text
7. **Data Loaders**: Create PyTorch data loaders
8. **Loss Calculation**: Baseline loss measurement
9. **Training Code**: Main training loop execution
10. **Generation with Temperature**: Advanced text generation
11. **Dependency Versions**: Environment verification
12. **GPT-2 Weight Loading**: Load pre-trained weights
13. **Pre-trained Model**: Create model with pre-trained weights
14. **Final Text Generation**: Test pre-trained model performance

### Training Features

- **AdamW Optimizer**: With weight decay for regularization
- **Learning Rate**: 5e-4 (configurable)
- **Evaluation**: Periodic validation during training
- **Checkpointing**: Model state preservation
- **Loss Tracking**: Both training and validation loss
- **Token Counting**: Track tokens processed during training
- **Text Sampling**: Generate samples during training for monitoring

## 🔧 Configuration

### Model Variants

The project supports multiple GPT model sizes:

```python
MODEL_CONFIGS = {
    "gpt2-small":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},  # 124M params
    "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}, # 355M params
    "gpt2-large":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20}, # 774M params
    "gpt2-xl":     {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}, # 1.5B params
}
```

### Training Parameters

```python
# Training configuration
num_epochs = 10           # Number of training epochs
batch_size = 2            # Batch size for training
learning_rate = 5e-4      # AdamW learning rate
weight_decay = 0.1        # Weight decay for regularization
eval_freq = 5             # Evaluate every N steps
eval_iter = 1             # Number of evaluation iterations
```

### Text Generation Parameters

```python
# Generation settings
max_new_tokens = 25       # Maximum tokens to generate
temperature = 1.5         # Sampling temperature (higher = more random)
top_k = 50               # Top-k sampling (keep top k tokens)
context_size = 256        # Context window size
```

## 🛠️ Development

### Adding New Features

1. **Model Components**: Add new layers in `src/model/` or `src/transformer/`
2. **Training Features**: Extend training utilities in `src/utils/`
3. **Dashboard Elements**: Modify `src/ui/` for new visualizations
4. **Data Processing**: Add data utilities in `src/utils/`

### Code Style

The project follows these conventions:
- Python 3.8+ features
- Type hints where helpful
- Docstrings for major functions
- Modular design with clear separation of concerns
- Config-driven architecture for flexibility

### Testing

Run tests for individual components:

```bash
# Test model forward pass
python -c "from src.model import GPTModel; print('Model import successful')"

# Test training utilities
python -c "from src.utils.train import train_model_simple; print('Training utilities OK')"

# Test web dashboard
python -c "from src.ui.web_app import create_app; print('Dashboard OK')"
```

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure you're in the project root and using the module syntax
python -m src.main  # ✅ Correct
python src/main.py  # ❌ May cause import issues
```

**2. GPU Memory Issues**
```python
# Reduce batch size or context length
batch_size = 1        # Smaller batch
ctx_len = 128        # Shorter context
```

**3. Dashboard Not Loading**
```bash
# Check Flask installation
pip install flask

# Verify port is available (5000)
netstat -an | grep 5000
```

**4. TensorFlow Warnings**
```bash
# TensorFlow is optional - warnings can be ignored
# Or install for additional features:
pip install tensorflow
```

### Performance Tips

1. **Use GPU**: Enable CUDA for faster training
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

2. **Optimize Data Loading**: Use multiple workers
   ```python
   train_loader = DataLoader(dataset, num_workers=4, pin_memory=True)
   ```

3. **Mixed Precision**: For newer GPUs
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

4. **Batch Size**: Start small and increase based on available memory

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sebastian Raschka** for the excellent [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) reference
- **OpenAI** for the GPT architecture and pre-trained weights
- **Hugging Face** for the transformers library and tokenizers
- **PyTorch** team for the deep learning framework

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement:

- Additional model architectures (BERT, T5, etc.)
- More sophisticated training techniques
- Enhanced dashboard features
- Performance optimizations
- Additional text generation methods
- Better documentation and examples

## 📬 Contact

If you have questions or suggestions, please open an issue on GitHub.

---

**Happy Training! 🚀**