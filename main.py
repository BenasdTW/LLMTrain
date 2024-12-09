from transformers import LlamaForCausalLM, LlamaTokenizer

model_name = "meta-llama/Llama-3.2-1B"  # Replace with the actual model path if hosted locally
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

from datasets import load_dataset

dataset = load_dataset("your_dataset_name")  # Replace with your dataset
# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()

from torch.utils.data import DataLoader
import torch

# train_loader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True)
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# model.train()
# for epoch in range(3):  # 3 epochs
#     for batch in train_loader:
#         inputs = batch["input_ids"].to("cuda")
#         labels = batch["labels"].to("cuda")

#         outputs = model(inputs, labels=labels)
#         loss = outputs.loss
#         loss.backward()

#         optimizer.step()
#         optimizer.zero_grad()

model.eval()
eval_loader = DataLoader(tokenized_datasets["test"], batch_size=8)

for batch in eval_loader:
    inputs = batch["input_ids"].to("cuda")
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits
