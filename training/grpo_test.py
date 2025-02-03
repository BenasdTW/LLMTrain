import sys
import os
import math
# Append the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# train_grpo.py
import torch
import bitsandbytes as bnb
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from configs_and_helpers import quantization_config, lora_config
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers.trainer_utils import IntervalStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
from peft import LoraConfig
from peft.optimizers import create_loraplus_optimizer

output_name = "grpo_test"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
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

model = get_peft_model(model, lora_config)

optimizer = create_loraplus_optimizer(
    model=model,
    optimizer_cls=torch.optim.AdamW,
    # optimizer_cls=bnb.optim.PagedAdamW8bit,
    lr=4e-5,
    eps=1e-6,
    betas=(0.9, 0.999),
    weight_decay=0.0,
    loraplus_lr_ratio=8,
)
scheduler = None

# tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenizer.pad_token_id = 151643
# print(f"{tokenizer.eos_token_id=}")
# print(f"{tokenizer.pad_token_id=}")

dataset = load_dataset("trl-lib/tldr", split="train[:1%]")
print(dataset)

def filter_func(example):
    return True if len(example["prompt"]) < 1000 else False
dataset = dataset.filter(filter_func, num_proc=16)
dataset, eval_set = torch.utils.data.random_split(dataset, [218, 16])
print(dataset)

# Define the reward function, which rewards completions that are close to 20 characters
def reward_func(length):
    target = 20
    max_reward = 100
    decade = 0.02
    diff = abs(target - length)
    r = max_reward * math.exp(-decade * diff)
    return r

def reward_len(completions, **kwargs):
    reward = [reward_func(len(completion.split())) for completion in completions]
    print(f"{completions=}")
    print(f"{reward=}")
    return reward

training_args = GRPOConfig(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    # per_device_eval_batch_size=1,
    # eval_accumulation_steps=4,
    num_train_epochs=3,
    num_generations=4,
    learning_rate=5e-5,
    use_liger_kernel=True,
    # logging_dir=f"./profile/{output_name}",
    # logging_steps=1,
    # save_strategy=IntervalStrategy.EPOCH,
    # eval_strategy=IntervalStrategy.STEPS,
    # eval_steps=8,
    # eval_on_start=True,
    # save_strategy=IntervalStrategy.EPOCH,
    # save_total_limit=1,
    bf16=True,
    # report_to="tensorboard",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

trainer = GRPOTrainer(
    model=model,
    # peft_config=lora_config,
    processing_class=tokenizer,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
    # eval_dataset=eval_set,
    optimizers=(optimizer, scheduler)
)
trainer.train()

trainer.model.save_pretrained(output_name)
tokenizer.save_pretrained(output_name)
