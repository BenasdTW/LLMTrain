import re
import torch
import json
from datetime import datetime
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers.trainer_utils import IntervalStrategy
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from liger_kernel.transformers import AutoLigerKernelForCausalLM

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = "<|finetune_right_pad_id|>"
print(f"{tokenizer.eos_token_id=}")
print(f"{tokenizer.pad_token_id=}")

# model_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# {schema}

# -- {system}
# <|eot_id|>
# <|start_header_id|>user<|end_header_id|>

# {question}<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>

# {query}<|end_of_text|>"""
model_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{schema}

-- {system}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""

# output_template = "{query}<|end_of_text|>"

def split_string(input_string):
    # Split using '--', allowing optional newlines (\n) before/after
    parts = re.split(r"\s*\n?\s*--\s*\n?\s*", input_string.strip())
    # Filter out empty strings
    return [part.strip() for part in parts if part.strip()]

def tokenize(inst, output, max_length):
    split = split_string(inst)
    if len(split) != 3:
        # 26 fails
        return None, None
    input_str = model_template.format(schema=split[0], system=split[1], question=split[2])
    instruction_len = len(torch.tensor(tokenizer.encode(input_str), dtype=torch.int64))

    input_str = input_str + output + tokenizer.eos_token
    model_inputs = tokenizer(input_str, max_length=max_length, padding="max_length", return_tensors="pt")
    return model_inputs, instruction_len
    
def filter_by_token_count(x, max_length):
    # Tokenize the text and check the length
    model_inputs, _ = tokenize(x["instruction"], x["output"], max_length)
    if model_inputs is None:
        return False
    return len(model_inputs["input_ids"].squeeze(0)) <= max_length

class NSText2SQLDataset(Dataset):
    def __init__(self, size=None, max_length=2048, split="train"):
        dataset = load_dataset("NumbersStation/NSText2SQL", split=split)
        self.dataset = dataset.filter(lambda x: filter_by_token_count(x, max_length))
        print(len(self.dataset))
        if size:
            self.dataset = self.dataset.select(range(size))
        self.max_length = max_length

        self.eos_token = tokenizer.eos_token
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        model_inputs, instruction_len = tokenize(self.dataset[index]["instruction"], self.dataset[index]["output"], self.max_length)

        model_inputs["input_ids"] = model_inputs["input_ids"].squeeze(0)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0)
        labels = model_inputs["input_ids"].clone()  # Labels for causal LM
        # Mask out the instruction string
        labels[:instruction_len] = -100
        # Mask out the paddings
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
            
        return model_inputs

# 1B: 1:55:43, 3B: 5:05:03
dataset = NSText2SQLDataset(size=102500, max_length=512)
dataset, eval_set = torch.utils.data.random_split(dataset, [102400, 100])
print(f"{len(dataset)=} {len(eval_set)=}")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
# Load the Base Model
model = AutoLigerKernelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    use_cache=False,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,  # Match input type
)

# Configure QLoRA with PEFT
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


# Define Training Arguments
# Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
#     = 256
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=4,
    eval_accumulation_steps=4,
    # torch_empty_cache_steps=1,
    learning_rate=2e-4,
    num_train_epochs=1,
    # use_liger_kernel=True,
    logging_dir="./logs",
    logging_steps=1,
    save_strategy=IntervalStrategy.EPOCH,
    eval_strategy=IntervalStrategy.STEPS,
    eval_steps=4,
    eval_on_start=True,
    save_total_limit=1,
    bf16=True,
    optim="adamw_torch",
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_set,
    tokenizer=tokenizer,
)


# Start training
trainer.model.print_trainable_parameters()
trainer.train()

output_dir = "text2sql-1b-Instruct-format"
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


# Format it as "MMDD-hhmmss"
now = datetime.now()
formatted_time = now.strftime("%m%d-%H%M%S")

print(f"Saved log to ./logs/{formatted_time}-trainer.log")

with open(f"./logs/{formatted_time}-trainer.log", "w") as f:
    for obj in trainer.state.log_history:
        obj_str = json.dumps(obj)
        f.write(obj_str)
        f.write("\n")

