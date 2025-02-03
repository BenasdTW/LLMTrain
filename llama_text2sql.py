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
from configs_and_helpers import quantization_config, lora_config_builder, loraplus_optimizer_builder, training_args_builder

output_name = "text2sql_test"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = "<|finetune_right_pad_id|>"
print(f"{tokenizer.eos_token_id=}")
print(f"{tokenizer.pad_token_id=}")

# max_length = 512: 220742
# dataset = NSText2SQLDataset(tokenizer=tokenizer, size=102500, max_length=512)
dataset = NSText2SQLDatasetFormatted(tokenizer=tokenizer, size=21504, max_length=512)
dataset, eval_set = torch.utils.data.random_split(dataset, [20480, 1024])
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

model = get_peft_model(model, lora_config_builder(modules_to_save=None))
print(f"{model=}")

optim = loraplus_optimizer_builder(model, lr=2e-4)

training_args = training_args_builder(output_name, eff_batch=128, device_batch=4, epochs=3)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_set,
    tokenizer=tokenizer,
    optimizers=optim,
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


