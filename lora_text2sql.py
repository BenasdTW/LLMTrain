import torch
from datetime import datetime
from trl import SFTTrainer
from peft import LoraConfig
from transformers.trainer_utils import IntervalStrategy
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from RefinedText2SQL import NSText2SQLDataset, NSText2SQLDatasetFormatted, CustomLoggingCallback

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = "<|finetune_right_pad_id|>"
print(f"{tokenizer.eos_token_id=}")
print(f"{tokenizer.pad_token_id=}")

# max_length = 512: 220742
# dataset = NSText2SQLDataset(tokenizer=tokenizer, size=102500, max_length=512)
dataset = NSText2SQLDatasetFormatted(tokenizer=tokenizer, size=102500, max_length=512)
dataset, eval_set = torch.utils.data.random_split(dataset, [102400, 100])
print(f"{len(dataset)=} {len(eval_set)=}")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
# Load the Base Model
model = AutoLigerKernelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    use_cache=False,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,  # Match input type
)

# Configure QLoRA with PEFT
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


# 1B: 1:55:43, 3B: 4:59:29, 8B: 10:29:01
# Define Training Arguments
# Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
#     = 256
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=8,
    eval_accumulation_steps=2,
    # torch_empty_cache_steps=1,
    learning_rate=2e-4,
    num_train_epochs=1,
    # use_liger_kernel=True,
    logging_dir="./logs",
    logging_steps=1,
    save_strategy=IntervalStrategy.EPOCH,
    eval_strategy=IntervalStrategy.STEPS,
    eval_steps=4,
    eval_on_start=True,
    save_total_limit=1,
    bf16=True,
    optim="adamw_torch",
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_set,
    tokenizer=tokenizer,
)

# Format it as "MMDD-hhmmss"
now = datetime.now()
formatted_time = now.strftime("%m%d-%H%M%S")

output_name = "test"
output_log = f"./logs/{output_name}-{formatted_time}.log"


# Start training
trainer.model.print_trainable_parameters()
trainer.add_callback(CustomLoggingCallback(output_log))
trainer.train()

trainer.model.save_pretrained(output_name)
tokenizer.save_pretrained(output_name)


