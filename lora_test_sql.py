import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset, load_dataset

# Step 1: 加載預訓練模型和標籤器
model_name = "meta-llama/Llama-3.2-1B"  # 替換為所需模型
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

wikisql = load_dataset("wikisql")

# Tokenize data
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=64
    )
    tokenized["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in ids]
        for ids in tokenized["input_ids"]
    ]
    return tokenized

def preprocess(example):
    question = example["question"]
    print(question)
    # print(example["sql"])
    ground_truth_sql = example["sql"][0]["human_readable"]
    ground_truth_sql = ground_truth_sql.strip()
    
    # Generate SQL query using the model
    prompt = f"You are a Text-to-SQL model. Please directly translate this question to SQL query. Do not explain anything. Question: {question}"
    tokenized = tokenizer(prompt, text_target=ground_truth_sql, truncation=True)
    return tokenized

train_data = wikisql["train"]
valid_data = wikisql["validation"]

valid_data = valid_data.map(preprocess, batched=True)
train_data = train_data.map(preprocess, batched=True)

# Step 2: 定義 LoRA 配置
lora_config = LoraConfig(
    r=16,                      # LoRA rank
    lora_alpha=32,             # LoRA alpha
    inference_mode=False,
    lora_dropout=0.1,          # Dropout 機率
    task_type=TaskType.CAUSAL_LM, # 設定任務類型 (因果語言建模)
)

# 將 LoRA 應用於模型
model = get_peft_model(model, lora_config)

# Step 4: Print Trainable Parameters
model.print_trainable_parameters()

# Step 4: 定義訓練參數
training_args = TrainingArguments(
    output_dir="./lora_text2sql_llama",
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    # fp16=True,  # 如果 GPU 支持，可以啟用混合精度訓練
    push_to_hub=False,
)

# Step 5: 定義 Trainer 並啟動訓練
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data,
)

trainer.train()

# Step 6: 儲存微調後的模型
model.save_pretrained("./lora_text2sql_llama")
tokenizer.save_pretrained("./lora_text2sql_llama")
