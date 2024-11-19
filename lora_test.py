import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from datasets import Dataset, load_dataset

# Step 1: 加載預訓練模型和標籤器
model_name = "gpt2"  # 替換為所需模型
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 確保標籤器支持自動填充
tokenizer.pad_token = tokenizer.eos_token

# Step 2: 定義 LoRA 配置
lora_config = LoraConfig(
    r=8,                      # LoRA rank
    lora_alpha=32,            # LoRA alpha
    # target_modules=["c_attn"], # 需要應用 LoRA 的模組，例如 GPT 中的注意力層
    inference_mode=False,
    lora_dropout=0.1,         # Dropout 機率
    # bias="none",              # 不對偏置項進行微調
    task_type=TaskType.CAUSAL_LM, # 設定任務類型 (因果語言建模)
)

# 將 LoRA 應用於模型
model = get_peft_model(model, lora_config)

# Step 4: Print Trainable Parameters
model.print_trainable_parameters()

# Step 3: 加載訓練數據
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_data = dataset["train"]
valid_data = dataset["validation"]

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

train_data = train_data.map(tokenize_function, batched=True)
# tokenized_dataset = tokenized_dataset.remove_columns(["text"]).with_format("torch")
valid_data = valid_data.map(tokenize_function, batched=True)

# Step 4: 定義訓練參數
training_args = TrainingArguments(
    output_dir="./lora_gpt2",
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
    fp16=True,  # 如果 GPU 支持，可以啟用混合精度訓練
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
model.save_pretrained("./lora_gpt2")
tokenizer.save_pretrained("./lora_gpt2")
