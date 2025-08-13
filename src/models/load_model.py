from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from ..config import LOCAL_LLM_MODEL_ID
from ..config import DEVICE_TYPE
# hf_login.py
import os
from dotenv import load_dotenv
from huggingface_hub import login
import torch

def load_local_llm(device_type: str = "cpu"):
    """
    Loads a local LLM with conditional quantization.
    
    - If device_type is 'cuda' and GPU is available → use 8-bit quantization with bitsandbytes.
    - Otherwise → load in FP32 for CPU.

    Args:
        model_id (str): Hugging Face model ID or local path.
        device_type (str): 'cpu' or 'cuda'.

    Returns:
        transformers.Pipeline: Text-generation pipeline.
    """
    # Force CPU if requested or if CUDA unavailable

    # Load environment variables from .env
    load_dotenv()

    # Get token
    hf_token = os.getenv("HF_API_KEY")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file.")

    # Login
    login(token=hf_token)
    print("Hugging Face authentication successful!")

    use_gpu = DEVICE_TYPE == "cuda" and torch.cuda.is_available()
    actual_device = "cuda" if use_gpu else "cpu"

    print(f"Loading {LOCAL_LLM_MODEL_ID} model on {actual_device}...")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL_ID)

    if use_gpu:
        # GPU path with bitsandbytes 8-bit quantization
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )

        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM_MODEL_ID,
            quantization_config=quant_config,
            device_map=actual_device,
            low_cpu_mem_usage=True
        )

    else:
        # CPU path — standard FP32
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM_MODEL_ID,
            device_map=actual_device,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        torch_dtype=torch.float32 if not use_gpu else None,
        device_map=actual_device
    )