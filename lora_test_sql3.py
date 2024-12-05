import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftConfig
from transformers.trainer_utils import IntervalStrategy

# Load the Dataset
dataset = load_dataset("spider", split="train")  # Example: Spider text-to-SQL dataset
test_dataset = load_dataset("spider", split="validation")

# model_template = """<|start_header_id|>system<|end_header_id|>

# You are a text-to-SQL model. Please generate a SQL query command according to the user input. Only generates the SQL query, do not explain anything.
# This is the SQL schema:
# {{ .Schema }}
# <|eot_id|>
# <|start_header_id|>user<|end_header_id|>

# {{ .TextQuery }}<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>

# {{ .SQLQuery }}
# """
# Preprocess the Dataset
def preprocess_function(examples):
    prompts = []
    for question, query in zip(examples["question"], examples["query"]):
        
        model_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a text-to-SQL model. Please generate a SQL query command according to the user input. Only generates the SQL query, do not explain anything.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{query}<|end_of_text|>"""
        prompts.append(model_template)
    # prompts = ["Generate an SQL query for the given question: " + question for question in examples["question"]]
    # targets = examples["query"]
    model_inputs = tokenizer(prompts, truncation=True, padding="max_length", max_length=256)
    model_inputs["labels"] = model_inputs["input_ids"].copy()  # Labels for causal LM
    # labels = tokenizer(targets, truncation=True, padding="max_length", max_length=256)["input_ids"]
    # model_inputs["labels"] = labels
    return model_inputs


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Load the Base Model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", 
    device_map="auto", 
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,  # Match input type
    bnb_4bit_compute_dtype=torch.bfloat16,  # Set compute dtype to float16
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

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy=IntervalStrategy.EPOCH,
    eval_strategy=IntervalStrategy.EPOCH,
    save_total_limit=1,
    # fp16=True,
    bf16=True,
    optim="adamw_torch",
    report_to="none"
)

# Define the Trainer
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
)

# Train the Model
trainer.train()

# Save the Model
model.save_pretrained("./finetuned-llama-text2sql")
tokenizer.save_pretrained("./finetuned-llama-text2sql")
