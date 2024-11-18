import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from datasets import Dataset

# Step 1: Load a Pretrained Model and Tokenizer
model_name = "gpt2"  # Replace with your model of choice
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1
)
# Ensure correct GPU is selected
device = torch.device("cuda")
# Step 3: Apply LoRA to the Model
peft_model = get_peft_model(model, lora_config)
peft_model = peft_model.to(device)
peft_model.print_trainable_parameters()

# Step 4: Prepare Dataset
# Example dataset with a single text column
data = {"text": ["Once upon a time, there was a brave knight.",
                 "In a galaxy far away, a hero emerged."]}
dataset = Dataset.from_dict(data)

# Tokenize data
def preprocess_function(examples):
    tokenized = tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=64
    )
    tokenized["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in ids]
        for ids in tokenized["input_ids"]
    ]
    return tokenized

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"]).with_format("torch")

# Step 5: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./lora_finetuned_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="no",
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01
)

# Step 6: Initialize Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Step 7: Train the Model
trainer.train()

# Step 8: Save the PEFT Model
peft_model.save_pretrained("lora_finetuned_model")

# Step 9: Load the Model for Inference
from peft import PeftModel
peft_model = PeftModel.from_pretrained(model, "lora_finetuned_model")
peft_model.eval()

# Tokenize the input with attention_mask
tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
inputs = tokenizer("Tell me a story about", return_tensors="pt", padding=True, truncation=True)

# Move inputs to the same device as the model
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
print("checkpoint 3")
print(f"Pad token ID: {tokenizer.pad_token_id}, EOS token ID: {tokenizer.eos_token_id}")
print(f"Input IDs: {input_ids}, Attention Mask: {attention_mask}")
outputs = peft_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
