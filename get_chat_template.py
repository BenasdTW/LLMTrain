import torch
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "System prompt test"},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]
print(tokenizer.apply_chat_template(messages, tokenize=False))