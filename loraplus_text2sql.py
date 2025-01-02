import torch
import bitsandbytes as bnb
from datetime import datetime
from trl import SFTTrainer
from peft import get_peft_model
from peft.optimizers import create_loraplus_optimizer
from transformers import AutoTokenizer
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from RefinedText2SQL import NSText2SQLDataset, NSText2SQLDatasetFormatted, CustomLoggingCallback
from transformers.trainer_utils import IntervalStrategy
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from lora_config import quantization_config, lora_config

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = "<|finetune_right_pad_id|>"
print(f"{tokenizer.eos_token_id=}")
print(f"{tokenizer.pad_token_id=}")

# max_length = 512: 220742
# dataset = NSText2SQLDataset(tokenizer=tokenizer, size=102500, max_length=512)
dataset = NSText2SQLDatasetFormatted(tokenizer=tokenizer, size=205800, max_length=512)
dataset, eval_set = torch.utils.data.random_split(dataset, [204800, 1000])
print(f"{len(dataset)=} {len(eval_set)=}")

# Load the Base Model
model = AutoLigerKernelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    use_cache=False,
    attn_implementation="flash_attention_2",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,  # Match input type
)

model = get_peft_model(model, lora_config)

optimizer = create_loraplus_optimizer(
    model=model,
    optimizer_cls=bnb.optim.PagedAdamW8bit,
    # optimizer_cls=torch.optim.AdamW,
    lr=5e-5,
    eps=1e-6,
    # eps=1e-8,
    betas=(0.9, 0.999),
    weight_decay=0.0,
    loraplus_lr_ratio=16,
)
scheduler = None

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=8,
    eval_accumulation_steps=4,
    # torch_empty_cache_steps=1,
    num_train_epochs=3,
    # use_liger_kernel=True,
    logging_dir="./profile",
    logging_steps=1,
    save_strategy=IntervalStrategy.EPOCH,
    eval_strategy=IntervalStrategy.STEPS,
    eval_steps=8,
    eval_on_start=True,
    save_total_limit=1,
    bf16=True,
    report_to="tensorboard",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

trainer = SFTTrainer(
    model=model,
    # peft_config=lora_config,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_set,
    tokenizer=tokenizer,
    optimizers=(optimizer, scheduler)
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


