# accelerate launch finetune_example_fsdp.py
import re
import torch
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from configs_and_helpers import quantization_config, lora_config_builder, loraplus_optimizer_builder, training_args_builder
from datasets import load_dataset
from accelerate import PartialState

output_name = "t"
model_name = "perplexity-ai/r1-1776-distill-llama-70b"

bnb_conf = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    # For FSDP
    bnb_4bit_quant_storage=torch.bfloat16,
    llm_int8_skip_modules=[
        "lm_head",
        "multi_modal_projector",
        "merger",
        "modality_projection",
        "model.layers.0.self_attn",
        "model.layers.0.mlp",
        "model.layers.3.mlp",
        "model.layers.55.self_attn"
    ],
)

# Load the base model (use liger kernel)
model = AutoModelForCausalLM.from_pretrained(
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_conf,  # QLoRA (NF4, DQ)
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer = AutoProcessor.from_pretrained(model_name)

# Save the LoRA adapter model
model.save_pretrained(output_name)
tokenizer.save_pretrained(output_name)


