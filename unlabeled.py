from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from datasets import load_dataset
from torchviz import make_dot
import torch


# Load model and tokenizer
model_name = "Llama-3.2-1B"  # or local path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)
print("Model Loaded")
print(model)
# x = torch.randn(1, 8)
x = torch.randint(low=0, high=100, size=(50000, 2048), dtype=torch.int)
y = model(x)
# Extract logits (actual tensor data)
logits = y.logits
# Generate and display the diagram
dot = make_dot(logits, params=dict(model.named_parameters()))
dot.render("neural_network", format="png")  # Saves the diagram as a PNG

# Load dataset (unlabeled text data)
# dataset = load_dataset("wikitext", "wikitext-103-v1")  # Example of raw text data
# Load the 'test' and 'validation' splits only
# test_dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
# validation_dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")


# # print(dataset)
# # print("First example in training split:", dataset['test'][:])
# print("Dataset Loaded")
# # Tokenize data
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

# # Set pad_token as eos_token
# tokenizer.pad_token = tokenizer.eos_token

# # Apply the tokenization to the test and validation datasets
# tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
# tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True)
# # tokenized_datasets = dataset.map(tokenize_function, batched=True)

# # Fine-tune with Causal Language Modeling
# from transformers import Trainer, TrainingArguments

# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=1,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_test_dataset,
#     eval_dataset=tokenized_validation_dataset,
# )

# trainer.train()
