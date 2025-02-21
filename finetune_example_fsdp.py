# accelerate launch finetune_example_fsdp.py
import re
import torch
from trl import SFTTrainer
from peft import get_peft_model
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs_and_helpers import quantization_config, lora_config_builder, loraplus_optimizer_builder, training_args_builder
from datasets import load_dataset
from accelerate import Accelerator

# accelerator = Accelerator()

output_name = "test"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
dataset_name = "NumbersStation/NSText2SQL"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the pad token to the bos token (151643), make sure to check the tokenizer_config.json of the base model
tokenizer.pad_token = tokenizer.bos_token
tokenizer.pad_token_id = 151643
print(f"{tokenizer.eos_token_id=}")
print(f"{tokenizer.pad_token_id=}")

# max_length = 512: 220742
max_length = 512

def split_string(input_string):
    # Split using '--', allowing optional newlines (\n) before/after
    parts = re.split(r"\s*\n?\s*--\s*\n?\s*", input_string.strip())
    # Filter out empty strings
    return [part.strip() for part in parts if part.strip()]

def format_and_tokenize(expample):
    split = split_string(expample["instruction"])
    # Filter out examples that don't follow the format
    if len(split) != 3:
        # 26 examples are not in the correct format
        return {"input_ids": None, "attention_mask": None, "labels": None}
    sys_prompt = f"{split[0]}\n\n-- {split[1]}"
    user_prompt = split[2]

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")

    input_str = input_str + expample["output"] + tokenizer.eos_token
    model_inputs = tokenizer(input_str, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")

    input_ids = model_inputs["input_ids"].squeeze(0)
    # assert len(input_ids) == max_length
    labels = input_ids.clone()
    # Label masking
    # Replace padding token id with -100 in labels
    labels[labels == tokenizer.pad_token_id] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": model_inputs["attention_mask"].squeeze(0),
        "labels": labels,
    }

    return out

# Filter out examples that don't follow the format
def filter_func(example):
    if example["input_ids"] is None:
        return False

    return True if example["labels"][-1] == -100 else False

# Preprocessing
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.map(format_and_tokenize, num_proc=16)

used_columns = ["input_ids", "attention_mask", "labels"]
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in used_columns])
dataset = dataset.filter(filter_func, num_proc=16)
dataset = dataset.select(range(21504))
# dataset, eval_set = torch.utils.data.random_split(dataset, [20480, 1024])
splits = dataset.train_test_split(test_size=1024)
dataset = splits["train"]
eval_set = splits["test"]

print(f"{dataset=}")
# Load the base model (use liger kernel)
model = AutoLigerKernelForCausalLM.from_pretrained(
# model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    # device_map="auto", 
    # device_map={"": accelerator.process_index},
    # device_map="cuda:0", 
    use_cache=False,
    attn_implementation="flash_attention_2",
    # quantization_config=quantization_config,  # QLoRA (NF4, DQ)
    torch_dtype=torch.bfloat16,  # Match input type
)

# Configure LoRA adapters
model = get_peft_model(model, lora_config_builder())

# LoRA+ Optimizer (ratio = 16)
optim = loraplus_optimizer_builder(model, lr=2e-4)

training_args = training_args_builder(output_name, eff_batch=128, device_batch=8, epochs=3)
# training_args.gradient_checkpointing_kwargs={"use_reentrant": True}
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_set,
    # optimizers=optim,
)

# Start training
trainer.model.print_trainable_parameters()
if getattr(trainer.accelerator.state, "fsdp_plugin", None):
    from peft.utils.other import fsdp_auto_wrap_policy
    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
trainer.train()

# Save the LoRA adapter model
trainer.model.save_pretrained(output_name)
tokenizer.save_pretrained(output_name)


