import sys
import os
# Append the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import torch
import bitsandbytes as bnb
from datetime import datetime
from trl import SFTTrainer
from peft import get_peft_model
from peft.optimizers import create_loraplus_optimizer
from transformers import AutoTokenizer
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from RefinedText2SQL import NSText2SQLDataset, NSText2SQLDatasetFormatted, CustomLoggingCallback
from transformers.trainer_utils import IntervalStrategy
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from configs_and_helpers import quantization_config, lora_config
from datasets import load_dataset

output_name = "loraplus0.5"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.bos_token
tokenizer.pad_token_id = 151643
print(f"{tokenizer.eos_token_id=}")
print(f"{tokenizer.pad_token_id=}")

# max_length = 512: 220742

def split_string(input_string):
    # Split using '--', allowing optional newlines (\n) before/after
    parts = re.split(r"\s*\n?\s*--\s*\n?\s*", input_string.strip())
    # Filter out empty strings
    return [part.strip() for part in parts if part.strip()]

max_length = 512
def format_and_tokenize(expample):
    split = split_string(expample["instruction"])
    if len(split) != 3:
        # 26 fails
        return {"input_ids": None, "attention_mask": None, "labels": None}
    sys_prompt = f"{split[0]}\n\n-- {split[1]}"
    user_prompt = split[2]

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")

    # instruction_len = len(torch.tensor(tokenizer.encode(input_str), dtype=torch.int32))

    input_str = input_str + expample["output"] + tokenizer.eos_token
    model_inputs = tokenizer(input_str, max_length=max_length, padding="max_length", return_tensors="pt")

    input_ids = model_inputs["input_ids"].squeeze(0)
    labels = input_ids.clone()
    # Replace padding token id with -100 in labels
    labels[labels == tokenizer.pad_token_id] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": model_inputs["attention_mask"].squeeze(0),
        "labels": labels,
    }

    # print(f"{out=}")
    return out

dataset = load_dataset("NumbersStation/NSText2SQL", split="train")

# Print each example in the subset
# for i, example in enumerate(subset_dataset):
#     print(f"Example {i}:")
#     print(example)
#     print("-" * 80)
def filter_func(example):
    if example["input_ids"] is None:
        return False

    return True if example["labels"][-1] == -100 else False

dataset = dataset.map(format_and_tokenize, num_proc=16)

used_columns = ["input_ids", "attention_mask", "labels"]
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in used_columns])
dataset = dataset.filter(filter_func, num_proc=16)
dataset = dataset.select(range(14080))
dataset, eval_set = torch.utils.data.random_split(dataset, [12800, 1280])

print(f"{dataset=}")
# Print each example in the subset
# for i, example in enumerate(dataset):
#     print(f"Example {i}:")
#     print(example)
#     print("=" * 80)

# dataset, eval_set = torch.utils.data.random_split(dataset, [204800, 1000])
# print(f"{len(dataset)=} {len(eval_set)=}")

# # Load the Base Model
model = AutoLigerKernelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    use_cache=False,
    attn_implementation="flash_attention_2",
    # quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,  # Match input type
)

model = get_peft_model(model, lora_config)

optimizer = create_loraplus_optimizer(
    model=model,
    # optimizer_cls=bnb.optim.PagedAdamW8bit,
    optimizer_cls=torch.optim.AdamW,
    lr=5e-5,
    eps=1e-8,
    betas=(0.9, 0.999),
    weight_decay=0.0,
    loraplus_lr_ratio=16,
)
scheduler = None

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=8,
    eval_accumulation_steps=16,
    # torch_empty_cache_steps=1,
    num_train_epochs=3,
    use_liger_kernel=True,
    logging_dir="./profile",
    logging_steps=1,
    save_strategy=IntervalStrategy.EPOCH,
    eval_strategy=IntervalStrategy.STEPS,
    eval_steps=10,
    eval_on_start=True,
    save_total_limit=1,
    bf16=True,
    report_to="tensorboard",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_set,
    processing_class=tokenizer,
    optimizers=(optimizer, scheduler)
)

# Format it as "MMDD-hhmmss"
now = datetime.now()
formatted_time = now.strftime("%m%d-%H%M%S")

output_log = f"./logs/{output_name}-{formatted_time}.log"


# Start training
trainer.model.print_trainable_parameters()
trainer.add_callback(CustomLoggingCallback(output_log))
trainer.train()

trainer.model.save_pretrained(output_name)
tokenizer.save_pretrained(output_name)


