from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"  
model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
print(f"{tokenizer.eos_token_id=}")
print(f"{tokenizer.pad_token_id=}")
print(f"{tokenizer.bos_token_id=}")

messages = [
    # {"role": "system", "content": "You are a friendly chatbot"},
    # {"role": "user", "content": "Hi there!"},
    {"role": "user", "content": "Caclulate 100 / (6/2)"},
]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))
outputs = model.generate(tokenized_chat.to(model.device), max_length=384)
print(outputs)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

