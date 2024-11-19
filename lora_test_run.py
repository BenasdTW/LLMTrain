import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Pretrained Model and Tokenizer
pretrained_model_name = 'gpt2'  # Change this to your pretrained model name
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
tokenizer.pad_token = tokenizer.eos_token
pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
pretrained_model.eval()

# Load Fine-Tuned Model and Tokenizer
fine_tuned_model_name = './lora_finetuned_model'  # Change to your fine-tuned model path
fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_name)
fine_tuned_model.eval()

# Tokenize the input
input_text = "Once upon a time, there was"
inputs = tokenizer(input_text, return_tensors="pt")

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model = pretrained_model.to(device)
fine_tuned_model = fine_tuned_model.to(device)

# Move inputs to the same device as the models
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Generate outputs from both models
with torch.no_grad():
    pretrained_outputs = pretrained_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
    fine_tuned_outputs = fine_tuned_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)

# Decode the generated outputs
pretrained_output_text = tokenizer.decode(pretrained_outputs[0], skip_special_tokens=True)
fine_tuned_output_text = tokenizer.decode(fine_tuned_outputs[0], skip_special_tokens=True)

# Print and Compare the outputs
print("Pretrained Model Output:", pretrained_output_text)
print("=" * 100)
print("Fine-Tuned Model Output:", fine_tuned_output_text)

# Optionally, compute some metric to compare the outputs (e.g., BLEU score, cosine similarity)
