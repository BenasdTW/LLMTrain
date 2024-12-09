import torch
import json
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers.trainer_utils import IntervalStrategy
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig

model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = "<|finetune_right_pad_id|>"
print(f"{tokenizer.eos_token_id=}")
print(f"{tokenizer.pad_token_id=}")

class NSText2SQLDataset(Dataset):
    def __init__(self, size=None, max_length=2048):
        self.dataset = load_dataset("NumbersStation/NSText2SQL",split="train")
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

dataset = NSText2SQLDataset(size=10, max_length=400)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
# Load the Base Model
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,  # Match input type
)

# Configure QLoRA with PEFT
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_dir="./logs",
    logging_steps=1,
    save_strategy=IntervalStrategy.EPOCH,
    # eval_strategy=IntervalStrategy.EPOCH,
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
    # eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
)


# Start training

trainer.model.print_trainable_parameters()
trainer.train()

output_dir = "test"
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

with open("trainer.log", "w") as f:
    for obj in trainer.state.log_history:
        obj_str = json.dumps(obj)
        f.write(obj_str)
        f.write("\n")


