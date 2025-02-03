from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"  
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

messages = [
    {"role": "system", "content": "You are a friendly chatbot"},
    {"role": "user", "content": "Hi there!"},
]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))

