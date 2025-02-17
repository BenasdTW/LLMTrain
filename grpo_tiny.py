import torch
from datasets import Dataset, load_dataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

# model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

model = AutoLigerKernelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# dummy_dataset = Dataset.from_dict({"text": ["Dummy dataset"] * 16, })
dataset = load_dataset("trl-lib/tldr", split="train[:10%]")
# train_dataset = load_dataset("trl-internal-testing/tiny-ultrafeedback-binarized", split="train")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(
    num_train_epochs=3,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    report_to="none",
    logging_steps=1,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",  # Target attention layers
    modules_to_save=["input_layernorm", "post_attention_layernorm", "norm"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
# model = get_peft_model(model, lora_config)
trainer = GRPOTrainer(
    model=model,
    peft_config=lora_config,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_funcs=reward_len,
)


trainer.train()
