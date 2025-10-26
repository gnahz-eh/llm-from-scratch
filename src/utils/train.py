import torch
import os
import requests
import json
import numpy as np

# Try to import TensorFlow, but make it optional
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Some functions may not work.")

from tqdm import tqdm
from .token import text_to_token_ids, token_ids_to_text
from .generate_text import generate_text_simple, generate
from tiktoken.load import load_tiktoken_bpe

def load_weights_from_hf_gpt2(model_size, device="cpu"):
    """
    Load GPT-2 weights from Hugging Face transformers (PyTorch alternative to TensorFlow)
    Returns parameters in the original TensorFlow checkpoint format for compatibility
    """
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        raise ImportError("transformers library is required. Install with: pip install transformers")
    
    # Map model sizes to Hugging Face model names
    model_mapping = {
        "124M": "openai-community/gpt2",
        "355M": "openai-community/gpt2-medium", 
        "774M": "openai-community/gpt2-large",
        "1558M": "openai-community/gpt2-xl"
    }
    
    if model_size not in model_mapping:
        raise ValueError(f"Model size {model_size} not supported. Choose from {list(model_mapping.keys())}")
    
    # Load the Hugging Face model
    hf_model = GPT2LMHeadModel.from_pretrained(model_mapping[model_size])
    hf_model.eval()
    
    # Extract state dict and convert to original TensorFlow format
    params = {}
    hf_state_dict = hf_model.state_dict()
    
    # Map Hugging Face parameter names to original TensorFlow checkpoint format
    for hf_name, tensor in hf_state_dict.items():
        if hf_name.startswith("transformer."):
            # Remove "transformer." prefix
            name = hf_name[len("transformer."):]
            
            if name == "wte.weight":
                params["wte"] = tensor.detach().numpy()
            elif name == "wpe.weight":
                params["wpe"] = tensor.detach().numpy()
            elif name == "ln_f.weight":
                params["g"] = tensor.detach().numpy()  # final layer norm weight (gain)
            elif name == "ln_f.bias":
                params["b"] = tensor.detach().numpy()  # final layer norm bias
            elif name.startswith("h."):
                # Layer-specific parameters - create nested structure
                if "blocks" not in params:
                    params["blocks"] = {}
                
                parts = name.split(".")
                layer_num = int(parts[1])
                param_name = ".".join(parts[2:])
                
                if layer_num not in params["blocks"]:
                    params["blocks"][layer_num] = {}
                
                # Create nested structure matching original TensorFlow format
                if param_name == "ln_1.weight":
                    if "ln_1" not in params["blocks"][layer_num]:
                        params["blocks"][layer_num]["ln_1"] = {}
                    params["blocks"][layer_num]["ln_1"]["g"] = tensor.detach().numpy()
                elif param_name == "ln_1.bias":
                    if "ln_1" not in params["blocks"][layer_num]:
                        params["blocks"][layer_num]["ln_1"] = {}
                    params["blocks"][layer_num]["ln_1"]["b"] = tensor.detach().numpy()
                elif param_name == "ln_2.weight":
                    if "ln_2" not in params["blocks"][layer_num]:
                        params["blocks"][layer_num]["ln_2"] = {}
                    params["blocks"][layer_num]["ln_2"]["g"] = tensor.detach().numpy()
                elif param_name == "ln_2.bias":
                    if "ln_2" not in params["blocks"][layer_num]:
                        params["blocks"][layer_num]["ln_2"] = {}
                    params["blocks"][layer_num]["ln_2"]["b"] = tensor.detach().numpy()
                elif param_name == "attn.c_attn.weight":
                    if "attn" not in params["blocks"][layer_num]:
                        params["blocks"][layer_num]["attn"] = {}
                    if "c_attn" not in params["blocks"][layer_num]["attn"]:
                        params["blocks"][layer_num]["attn"]["c_attn"] = {}
                    params["blocks"][layer_num]["attn"]["c_attn"]["w"] = tensor.detach().numpy()
                elif param_name == "attn.c_attn.bias":
                    if "attn" not in params["blocks"][layer_num]:
                        params["blocks"][layer_num]["attn"] = {}
                    if "c_attn" not in params["blocks"][layer_num]["attn"]:
                        params["blocks"][layer_num]["attn"]["c_attn"] = {}
                    params["blocks"][layer_num]["attn"]["c_attn"]["b"] = tensor.detach().numpy()
                elif param_name == "attn.c_proj.weight":
                    if "attn" not in params["blocks"][layer_num]:
                        params["blocks"][layer_num]["attn"] = {}
                    if "c_proj" not in params["blocks"][layer_num]["attn"]:
                        params["blocks"][layer_num]["attn"]["c_proj"] = {}
                    params["blocks"][layer_num]["attn"]["c_proj"]["w"] = tensor.detach().numpy()
                elif param_name == "attn.c_proj.bias":
                    if "attn" not in params["blocks"][layer_num]:
                        params["blocks"][layer_num]["attn"] = {}
                    if "c_proj" not in params["blocks"][layer_num]["attn"]:
                        params["blocks"][layer_num]["attn"]["c_proj"] = {}
                    params["blocks"][layer_num]["attn"]["c_proj"]["b"] = tensor.detach().numpy()
                elif param_name == "mlp.c_fc.weight":
                    if "mlp" not in params["blocks"][layer_num]:
                        params["blocks"][layer_num]["mlp"] = {}
                    if "c_fc" not in params["blocks"][layer_num]["mlp"]:
                        params["blocks"][layer_num]["mlp"]["c_fc"] = {}
                    params["blocks"][layer_num]["mlp"]["c_fc"]["w"] = tensor.detach().numpy()
                elif param_name == "mlp.c_fc.bias":
                    if "mlp" not in params["blocks"][layer_num]:
                        params["blocks"][layer_num]["mlp"] = {}
                    if "c_fc" not in params["blocks"][layer_num]["mlp"]:
                        params["blocks"][layer_num]["mlp"]["c_fc"] = {}
                    params["blocks"][layer_num]["mlp"]["c_fc"]["b"] = tensor.detach().numpy()
                elif param_name == "mlp.c_proj.weight":
                    if "mlp" not in params["blocks"][layer_num]:
                        params["blocks"][layer_num]["mlp"] = {}
                    if "c_proj" not in params["blocks"][layer_num]["mlp"]:
                        params["blocks"][layer_num]["mlp"]["c_proj"] = {}
                    params["blocks"][layer_num]["mlp"]["c_proj"]["w"] = tensor.detach().numpy()
                elif param_name == "mlp.c_proj.bias":
                    if "mlp" not in params["blocks"][layer_num]:
                        params["blocks"][layer_num]["mlp"] = {}
                    if "c_proj" not in params["blocks"][layer_num]["mlp"]:
                        params["blocks"][layer_num]["mlp"]["c_proj"] = {}
                    params["blocks"][layer_num]["mlp"]["c_proj"]["b"] = tensor.detach().numpy()
    
    return params

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    logits = model(input_batch)
    logits = logits.flatten(0, 1)
    loss = torch.nn.functional.cross_entropy(logits, target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None): # num_batches为计算损失的批次范围
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        # 取num_batches和len(data_loader)两者较小值以匹配data_loader中的总批次数量
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Try to import UI functions, but don't fail if not available
    try:
        from src.ui.web_app import log_training_epoch, log_message
        ui_available = True
    except ImportError:
        ui_available = False
    
    # 初始化列表以跟踪损失和已观察到的token
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    if ui_available:
        log_message(f"🚀 Starting training for {num_epochs} epochs")

    # 主要的训练步骤
    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式
        
        if ui_available:
            log_message(f"📈 Starting epoch {epoch + 1}/{num_epochs}")
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # 每个epoch开始之前重新设置梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # 计算损失梯度
            optimizer.step() # 利用损失梯度更新模型参数
            tokens_seen += input_batch.numel()
            global_step += 1

            # 可选的验证评估步骤
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
                # Update UI with training progress
                if ui_available:
                    log_training_epoch(epoch + 1, train_loss, val_loss, tokens_seen)

        # 在每个epoch完成后打印一个生成的文本示例
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
        
        if ui_available:
            log_message(f"✅ Completed epoch {epoch + 1}/{num_epochs}")

    if ui_available:
        log_message("🎉 Training completed successfully!", "success")

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()


def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch





def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    # Add backup URL
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"

    # Download files
    os.makedirs(model_dir, exist_ok=True)

    # Download and load model checkpoint
    checkpoint_path = os.path.join(model_dir, "checkpoint")
    checkpoint_url = os.path.join(base_url, model_size, "checkpoint")
    checkpoint_backup_url = os.path.join(backup_base_url, model_size, "checkpoint")
    download_file(checkpoint_url, checkpoint_path, checkpoint_backup_url)

    # Download and load settings
    hparams_path = os.path.join(model_dir, "hparams.json")
    hparams_url = os.path.join(base_url, model_size, "hparams.json")
    hparams_backup_url = os.path.join(backup_base_url, model_size, "hparams.json")
    download_file(hparams_url, hparams_path, hparams_backup_url)
    settings = json.load(open(hparams_path, "r"))

    # Download and load BPE vocabulary
    bpe_path = os.path.join(model_dir, "encoder.json")
    bpe_url = os.path.join(base_url, model_size, "encoder.json")
    bpe_backup_url = os.path.join(backup_base_url, model_size, "encoder.json")
    download_file(bpe_url, bpe_path, bpe_backup_url)

    merges_path = os.path.join(model_dir, "vocab.bpe")
    merges_url = os.path.join(base_url, model_size, "vocab.bpe")
    merges_backup_url = os.path.join(backup_base_url, model_size, "vocab.bpe")
    download_file(merges_url, merges_path, merges_backup_url)

    # Download model weights
    ckpt_path = os.path.join(model_dir, "model.ckpt")
    ckpt_url = os.path.join(base_url, model_size, "model.ckpt.index")
    ckpt_backup_url = os.path.join(backup_base_url, model_size, "model.ckpt.index")
    download_file(ckpt_url, f"{ckpt_path}.index", ckpt_backup_url)

    ckpt_meta_path = os.path.join(model_dir, "model.ckpt")
    ckpt_meta_url = os.path.join(base_url, model_size, "model.ckpt.meta")
    ckpt_backup_url = os.path.join(backup_base_url, model_size, "model.ckpt.meta")
    download_file(ckpt_meta_url, f"{ckpt_meta_path}.meta", ckpt_backup_url)

    ckpt_data_url = os.path.join(base_url, model_size, "model.ckpt.data-00000-of-00001")
    ckpt_data_backup_url = os.path.join(backup_base_url, model_size, "model.ckpt.data-00000-of-00001")
    download_file(ckpt_data_url, f"{ckpt_path}.data-00000-of-00001", ckpt_data_backup_url)

    # Load weights - try TensorFlow first, fallback to Hugging Face
    try:
        params = load_weights_from_tf_ckpt(f"{ckpt_path}.data-00000-of-00001", settings)
        print("Loaded weights from TensorFlow checkpoint")
    except ImportError:
        print("TensorFlow not available, using Hugging Face transformers to load weights...")
        params = load_weights_from_hf_gpt2(model_size)
        print("Loaded weights from Hugging Face model")

    # Load BPE tokenizer
    with open(bpe_path, "r", encoding="utf-8") as f:
        bpe_ranks = json.load(f)
    with open(merges_path, "r", encoding="utf-8") as f:
        merges = f.read().split("\n")[1:-1]
    
    tokenizer = {
        "bpe_ranks": bpe_ranks,
        "merges": merges
    }

    return settings, params, tokenizer


def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        file_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and has same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size and file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return True

        block_size = 1024  # 1 KB
        desc = os.path.basename(download_url)
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        return True

    try:
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException:
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except requests.exceptions.RequestException:
                pass

        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Alternative way using `requests`
"""
def download_file(url, destination):
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True)

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            # Iterate over the file data in chunks
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update progress bar
                file.write(chunk)  # Write the chunk to the file
"""

def load_weights_from_tf_ckpt(ckpt_path, settings):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to load weights from TensorFlow checkpoint. Please install TensorFlow or use an alternative method.")
    
    # Unused arguments, but there for consistency with other
    # load_* functions
    # del settings
    
    # Initialize parameters dictionary
    params = {}

    # Load the checkpoint
    reader = tf.train.load_checkpoint(ckpt_path)

    # Iterate over variables and map them
    for name, shape in reader.get_variable_to_shape_map().items():
        # Get the tensor value
        tensor = reader.get_tensor(name)

        # Get the variable name and remove the ".ATTRIBUTES" suffix
        name_without_suffix = name.replace("/.ATTRIBUTES/VARIABLE_VALUE", "")
        
        # Skip the global step variable
        if "global_step" in name_without_suffix:
            continue

        # Process the tensor based on its name
        if name_without_suffix.startswith("model/"):
            name_without_suffix = name_without_suffix[len("model/"):]
        
        if name_without_suffix.startswith("h"):
            layer_num = int(name_without_suffix.split("/")[0][1:])
            if name_without_suffix.endswith("/attn/c_attn/w"):
                params[f"layers.{layer_num}.att.W_query.weight"] = torch.tensor(tensor[:, :768])
                params[f"layers.{layer_num}.att.W_key.weight"] = torch.tensor(tensor[:, 768:1536])
                params[f"layers.{layer_num}.att.W_value.weight"] = torch.tensor(tensor[:, 1536:])
            elif name_without_suffix.endswith("/attn/c_proj/w"):
                params[f"layers.{layer_num}.att.out_proj.weight"] = torch.tensor(tensor)
            elif name_without_suffix.endswith("/mlp/c_fc/w"):
                params[f"layers.{layer_num}.ff.fc1.weight"] = torch.tensor(tensor)
            elif name_without_suffix.endswith("/mlp/c_proj/w"):
                params[f"layers.{layer_num}.ff.fc2.weight"] = torch.tensor(tensor)
            elif name_without_suffix.endswith("/ln_1/b"):
                params[f"layers.{layer_num}.norm1.bias"] = torch.tensor(tensor)
            elif name_without_suffix.endswith("/ln_1/g"):
                params[f"layers.{layer_num}.norm1.weight"] = torch.tensor(tensor)
            elif name_without_suffix.endswith("/ln_2/b"):
                params[f"layers.{layer_num}.norm2.bias"] = torch.tensor(tensor)
            elif name_without_suffix.endswith("/ln_2/g"):
                params[f"layers.{layer_num}.norm2.weight"] = torch.tensor(tensor)
            elif name_without_suffix.endswith("/attn/c_attn/b"):
                params[f"layers.{layer_num}.att.W_query.bias"] = torch.tensor(tensor[:768])
                params[f"layers.{layer_num}.att.W_key.bias"] = torch.tensor(tensor[768:1536])
                params[f"layers.{layer_num}.att.W_value.bias"] = torch.tensor(tensor[1536:])
            elif name_without_suffix.endswith("/attn/c_proj/b"):
                params[f"layers.{layer_num}.att.out_proj.bias"] = torch.tensor(tensor)
            elif name_without_suffix.endswith("/mlp/c_fc/b"):
                params[f"layers.{layer_num}.ff.fc1.bias"] = torch.tensor(tensor)
            elif name_without_suffix.endswith("/mlp/c_proj/b"):
                params[f"layers.{layer_num}.ff.fc2.bias"] = torch.tensor(tensor)
        elif name_without_suffix == "wpe":
            params["pos_emb.weight"] = torch.tensor(tensor)
        elif name_without_suffix == "wte":
            params["tok_emb.weight"] = torch.tensor(tensor)
        elif name_without_suffix == "ln_f/b":
            params["final_norm.bias"] = torch.tensor(tensor)
        elif name_without_suffix == "ln_f/g":
            params["final_norm.weight"] = torch.tensor(tensor)

    return params

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    # Use the recommended way to convert numpy array or tensor to torch.nn.Parameter
    if isinstance(right, np.ndarray):
        right_tensor = torch.from_numpy(right)
    else:
        right_tensor = right.detach().clone()
    return torch.nn.Parameter(right_tensor)

def load_weights_into_gpt(gpt, params):
    # Weight tying
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)
    
        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)
    
        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])
    
        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])
    
        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])
    
        gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
        gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])