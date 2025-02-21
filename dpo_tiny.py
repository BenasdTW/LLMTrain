import torch
from datasets import Dataset, load_dataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

# model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

model = AutoLigerKernelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# dummy_dataset = Dataset.from_dict({"text": ["Dummy dataset"] * 16, })
# train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
train_dataset = load_dataset("trl-internal-testing/tiny-ultrafeedback-binarized", split="train")

training_args = DPOConfig(
    num_train_epochs=10,
    per_device_train_batch_size=16,
    report_to="none",
    logging_steps=1,
    bf16=True,
    gradient_checkpointing=True,
)

# 37 ~ 46 GB
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

trainer.train()
