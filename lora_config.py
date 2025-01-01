import torch
from trl import SFTTrainer
from peft import LoraConfig
from transformers.trainer_utils import IntervalStrategy
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

# Configure QLoRA with PEFT
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention layers
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Target attention layers
    modules_to_save=["input_layernorm", "post_attention_layernorm", "norm"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


# 1B: 1:55:43, 3B: 4:59:29, 8B: 10:29:01
# Define Training Arguments
# Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
#     = 256
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=8,
    eval_accumulation_steps=4,
    # torch_empty_cache_steps=1,
    learning_rate=2e-4,
    num_train_epochs=3,
    # use_liger_kernel=True,
    logging_dir="./logs",
    logging_steps=1,
    save_strategy=IntervalStrategy.EPOCH,
    eval_strategy=IntervalStrategy.STEPS,
    eval_steps=8,
    eval_on_start=True,
    save_total_limit=1,
    bf16=True,
    optim="adamw_torch",
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)