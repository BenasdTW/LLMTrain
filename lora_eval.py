from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch

# Load the WikiSQL dataset and the fine-tuned model
wikisql = load_dataset("wikisql")
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # 替換為所需模型
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Function to generate SQL query for a given question
def generate_sql(question):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Set attention mask and pad_token_id
    attention_mask = inputs['attention_mask']
    pad_token_id = tokenizer.pad_token_id  # Fallback to eos_token_id if pad_token_id is None
    
    # Generate SQL query
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'], 
            attention_mask=attention_mask, 
            pad_token_id=pad_token_id, 
            max_length=256,  # Limit max length of generated query
            num_beams=3,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to compute Exact Match (EM) accuracy
def evaluate_exact_match(dataset):
    correct = 0
    total = 0
    for example in dataset:
        question = example["question"]
        ground_truth_sql = example["sql"]["human_readable"]
        ground_truth_sql = ground_truth_sql.strip()
        
        # Generate SQL query using the model
        prompt = f"You are a Text-to-SQL model. Please directly translate this question to SQL query. Do not explain anything. Question: {question}"
        generated_sql = generate_sql(prompt)
        # Remove string question from string gererated_sql
        generated_sql = generated_sql.replace(prompt, "").strip()

        print("=============================================")
        print("Question:")
        print(question)
        print("=============================================")
        print("Ground Truth:")
        print(ground_truth_sql)
        print("=============================================")
        print("Gererated:")
        print(generated_sql)
        if generated_sql == ground_truth_sql:
            correct += 1
        total += 1
        
    return correct / total

# Evaluate on the validation set
em_accuracy = evaluate_exact_match(wikisql['validation'])
print(f"Exact Match Accuracy: {em_accuracy * 100:.2f}%")
