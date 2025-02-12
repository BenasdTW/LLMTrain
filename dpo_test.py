# train_dpo.py
import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from configs_and_helpers import quantization_config, lora_config_builder
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers.trainer_utils import IntervalStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
from peft import LoraConfig
from peft.optimizers import create_loraplus_optimizer
from datasets import load_dataset

output_name = "DPO"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# The model to optimize
model = AutoLigerKernelForCausalLM.from_pretrained(
    model_name,
    device_map="auto", 
    attn_implementation="flash_attention_2",
    use_cache=False,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,  # Match input type
)

tokenizer.pad_token = tokenizer.bos_token
tokenizer.pad_token_id = 151643
print(f"{tokenizer.eos_token_id=}")
print(f"{tokenizer.pad_token_id=}")

# dataset = load_dataset("trl-lib/tldr", split="train[:1%]")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:10%]")
eval_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="test[:10%]")

training_args = DPOConfig(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    num_train_epochs=3,
    use_liger_kernel=True,
    logging_dir=f"./profile/{output_name}",
    logging_steps=1,
    save_strategy=IntervalStrategy.EPOCH,
    eval_strategy=IntervalStrategy.STEPS,
    eval_steps=10,
    save_total_limit=1,
    bf16=True,
    report_to="tensorboard",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
trainer = DPOTrainer(
    model=model,
    peft_config=lora_config_builder(),
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()

trainer.model.save_pretrained(f"./model/{output_name}")
tokenizer.save_pretrained(f"./model/{output_name}")
