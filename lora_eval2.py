from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch

# Load the WikiSQL dataset and the fine-tuned model
model_name = "meta-llama/Llama-3.2-1B"  # 替換為所需模型
# model_name = "./finetuned-llama-text2sql"  # 替換為所需模型
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Preprocess the Dataset
def preprocess_function(examples):
    prompts = ["Generate an SQL query for the given question: " + question for question in examples["question"]]
    targets = examples["query"]
    model_inputs = tokenizer(prompts, truncation=True, padding="max_length", max_length=256)
    labels = tokenizer(targets, truncation=True, padding="max_length", max_length=256)["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

test_dataset = load_dataset("spider", split="validation")
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

def generate_sql(example):
    # Convert input_ids and attention_mask to PyTorch tensors
    input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(model.device)
    attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(model.device)
    
    # Set pad_token_id (fallback to eos_token_id if None)
    pad_token_id = tokenizer.pad_token_id
    
    # Generate SQL query
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            max_length=512,  # Limit max length of generated query
            num_beams=3,
            early_stopping=True
        )
    
    return outputs[0]


# Function to compute Exact Match (EM) accuracy
def evaluate_exact_match(dataset):
    correct = 0
    total = 0
    for example in dataset:
        q_tokenized = example["input_ids"]
        # print(q_tokenized)
        ans_tokenized = example["labels"]
        # print(ans_tokenized)
        
        # Generate SQL query using the model
        generated_sql_token = generate_sql(example)
        question = tokenizer.decode(q_tokenized, skip_special_tokens=True)
        ans = tokenizer.decode(ans_tokenized, skip_special_tokens=True)
        # Remove string question from string gererated_sql
        generated_sql = tokenizer.decode(generated_sql_token, skip_special_tokens=True)
        # generated_sql = generated_sql.replace(question, "").strip()

        print("=============================================")
        print("Question:")
        print(question)
        print("=============================================")
        print("Ground Truth:")
        print(ans)
        print("=============================================")
        print("Gererated:")
        print(generated_sql)
        print(generated_sql_token)
        if generated_sql == ans:
            correct += 1
        total += 1
        
    return correct / total

# Evaluate on the validation set
em_accuracy = evaluate_exact_match(tokenized_test_dataset)
print(f"Exact Match Accuracy: {em_accuracy * 100:.2f}%")
