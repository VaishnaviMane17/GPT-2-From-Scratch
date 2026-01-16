# src/utils.py (Updated with additional utils)

import json
import urllib
import ssl
import os
import re
import torch
from typing import Dict, List, Any

def download_and_load_file(file_path: str, url: str) -> List[Dict]:
    """Download and load a JSON file."""
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url, context=ssl_context) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def format_input(entry: Dict) -> str:
    """Format instruction input for prompts."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

def get_base_config(model_name: str) -> Dict:
    """
    Get the base configuration for a GPT model based on the model name.
    """
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    config = BASE_CONFIG.copy()
    config.update(model_configs.get(model_name, {}))
    return config

def download_and_load_gpt2(model_size: str, models_dir: str = "gpt2") -> Tuple[Dict, Dict]:
    """
    Download and load GPT-2 settings and parameters.
    Placeholder implementation; adapt based on actual download logic.
    """
    
    settings = {}  # Load settings (e.g., from JSON)
    params = {}    # Load parameters (e.g., state_dict tensors)
    # Simulate download (replace with actual code)
    print(f"Downloading and loading GPT-2 {model_size} from {models_dir}...")
    # Actual implementation would fetch from OpenAI's model weights URLs or HF.
    return settings, params

def load_weights_into_gpt(model: torch.nn.Module, params: Dict[str, Any]) -> None:
    """
    Load pretrained weights into the GPT model.
    This maps the params dictionary to the model's state_dict.
    """
    # Get the model's current state dict
    model_state = model.state_dict()
    
    # Example mapping (adjust based on actual param keys)
    for name, param in params.items():
        if name.endswith(".attn.c_attn.w"):
            # Split QKV weights if combined
            qkv_dim = param.shape[0] // 3
            model_state[name.replace(".c_attn.w", ".W_query.weight")] = param[:qkv_dim]
            model_state[name.replace(".c_attn.w", ".W_key.weight")] = param[qkv_dim:2*qkv_dim]
            model_state[name.replace(".c_attn.w", ".W_value.weight")] = param[2*qkv_dim:]
        elif name in model_state:
            model_state[name] = torch.tensor(param)
        else:
            print(f"Warning: Param {name} not found in model state_dict.")
    
    model.load_state_dict(model_state)
    print("Pretrained weights loaded successfully.")