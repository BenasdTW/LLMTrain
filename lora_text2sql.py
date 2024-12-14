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

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = "<|finetune_right_pad_id|>"
print(f"{tokenizer.eos_token_id=}")
print(f"{tokenizer.pad_token_id=}")

class NSText2SQLDataset(Dataset):
    def __init__(self, size=None, max_length=2048, split="train"):
        self.dataset = load_dataset("NumbersStation/NSText2SQL", split=split)
        if size:
            self.dataset = self.dataset.select(range(size))
        self.max_length = max_length

        self.eos_token = tokenizer.eos_token
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        instruction_len = len(torch.tensor(tokenizer.encode(self.dataset[index]['instruction']), dtype=torch.int64))
        input_str = self.dataset[index]['instruction'] + self.dataset[index]["output"] + self.eos_token
        model_inputs = tokenizer(input_str, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        model_inputs["input_ids"] = model_inputs["input_ids"].squeeze(0)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0)
        labels = model_inputs["input_ids"].clone()  # Labels for causal LM
        # Mask out the instruction string
        labels[:instruction_len] = -100
        # Mask out the paddings
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
            
        return model_inputs

dataset = NSText2SQLDataset(size=100000, max_length=400)
dataset, eval_set = torch.utils.data.random_split(dataset, [0.99, 0.01])
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
#     = 128
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=20,
    # torch_empty_cache_steps=1,
    learning_rate=2e-4,
    num_train_epochs=1,
    # use_liger_kernel=True,
    logging_dir="./logs",
    logging_steps=1,
    save_strategy=IntervalStrategy.EPOCH,
    eval_strategy=IntervalStrategy.EPOCH,
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

output_dir = "test2"
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


