import re
import os
import torch
from transformers import BitsAndBytesConfig

MODEL_CONFIGS = {
    # --- LLaMA-4 Scout (107/16B) ----------------------------------
    r"^LLaMA-4-.*$": {
        "torch_dtype": torch.bfloat16,
        "quantize": False,
        "max_len": 8192,
    },

    # --- LLaMA-3 family ------------------------------------------
    r"^LLaMA-3-.*$": {
        "torch_dtype": torch.float16,
        "quantize": False,
        "max_len": 8192,
    },

    # --- LLaMA-2 family ------------------------------------------
    r"^LLaMA-2-.*$": {
        "torch_dtype": torch.float16,
        "quantize": False,
        "max_len": 4096,
    },

    # --- Gemma-3 series ------------------------------------------
    # 1 B size: 32K tokens total (text+images)
    r"^Gemma-3-1B$": {
        "torch_dtype": torch.bfloat16,
        "quantize": False,
        "max_len": 8192,   # 32 K tokens context
    },
    # 4 B, 12 B, 27 B sizes: 128K tokens total (text+images)
    r"^Gemma-3-(4B|12B)$": {
        "torch_dtype": torch.bfloat16,
        "quantize": False,
        "max_len": 8192,  # 128 K tokens context
    },
    r"^Gemma-3-27B$": {
        "torch_dtype": torch.bfloat16,
        "quantize": False,  # 8-bit quantization because of size
        "max_len": 8192,   # 128 K tokens context 131072           
    },

    # --- Qwen-3 Base models --------------------------------------
    r"^Qwen-3-.*-Base$": {
        "torch_dtype": torch.bfloat16,
        "quantize": False,
        "max_len": 8192,
    },
    r"^Qwen-3-32B$": {
        "torch_dtype": torch.bfloat16,
        "quantize": False,
        "max_len": 8192,
    },
    r"^Qwen-QwQ-32B$": {
        "torch_dtype": torch.bfloat16,
        "quantize": False,
        "max_len": 8192,
    },

    # --- DeepSeek family -----------------------------------------
    r"^DeepSeekMOE-16B$": {
        "torch_dtype": torch.bfloat16,
        "quantize": False,
        "max_len": 4096,
    },

    # --- Distilled Llama from DeepSeek-R1 -----------------------
    r"^DS-R1-Distill-Llama-8B$": {
        "torch_dtype": torch.float16,
        "quantize": False,
        "max_len": 8192,
    },
    r"^DS-R1-Distill-Llama-70B$": {
        "torch_dtype": torch.float16,
        "quantize": False,
        "max_len": 8192,
    },
    # Qwen3-30B-A3B
    r"^Qwen3-30B-A3B$": {
        "torch_dtype": torch.bfloat16,
        "quantize": False,
        "max_len": 8192,
        "bnb_config": None,
    },
    r"^Qwen-QwQ-32B$": {
        "torch_dtype": torch.bfloat16,
        "quantize": False,
        "max_len": 4096,
    },

}

FALLBACK = {
    "torch_dtype": torch.bfloat16,
    "quantize": False,
    "max_len": 4096,
    "bnb_config": None,
}


def get_model_cfg(model_path: str):
    # 1) strip directory → just the folder/name
    name = os.path.basename(model_path)
    # 2) remove an optional "-it" or "_it" suffix
    name = re.sub(r"[-_](?:it|pt)$", "", name, flags=re.IGNORECASE)

    for pattern, cfg in MODEL_CONFIGS.items():
        if re.match(pattern, name, flags=re.IGNORECASE):
            # ensure bnb_config if needed
            if cfg.get("quantize") and "bnb_config" not in cfg:
                cfg = cfg.copy()
                cfg["bnb_config"] = BitsAndBytesConfig(
                    load_in_4bit=False,
                    llm_int4_enable_fp32_cpu_offload=False,
                    offload_state_dict=False,
                    offload_folder=os.environ["JOB_OFFLOAD_DIR"]
                )
            return cfg
    return FALLBACK.copy()

