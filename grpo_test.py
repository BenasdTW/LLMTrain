# train_grpo.py
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from configs_and_helpers import quantization_config, lora_config
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers.trainer_utils import IntervalStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
from peft import LoraConfig
from peft.optimizers import create_loraplus_optimizer

output_name = "Qwen2.5-0.5B-GRPO"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# The model to optimize
model = AutoLigerKernelForCausalLM.from_pretrained(
    model_name,
    device_map="auto", 
    attn_implementation="flash_attention_2",
    use_cache=False,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,  # Match input type
)

tokenizer.pad_token = tokenizer.bos_token
tokenizer.pad_token_id = 151643
print(f"{tokenizer.eos_token_id=}")
print(f"{tokenizer.pad_token_id=}")

dataset = load_dataset("trl-lib/tldr", split="train[:1%]")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    # use_liger_kernel=True,
    logging_dir=f"./profile/{output_name}",
    logging_steps=1,
    save_strategy=IntervalStrategy.EPOCH,
    save_total_limit=1,
    bf16=True,
    report_to="tensorboard",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
trainer = GRPOTrainer(
    model=model,
    peft_config=lora_config,
    processing_class=tokenizer,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
