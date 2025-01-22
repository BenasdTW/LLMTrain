import torch
from datetime import datetime
from trl import SFTTrainer
from transformers import AutoTokenizer
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from RefinedText2SQL import NSText2SQLDataset, NSText2SQLDatasetFormatted, CustomLoggingCallback
from configs_and_helpers import quantization_config, lora_config, training_args


model_name = "meta-llama/Llama-3.2-1B-Instruct"
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
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,  # Match input type
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

output_name = "text2sql-1b-Instruct-less"
output_log = f"./logs/{output_name}-{formatted_time}.log"


# Start training
trainer.model.print_trainable_parameters()
trainer.add_callback(CustomLoggingCallback(output_log))
trainer.train()

trainer.model.save_pretrained(output_name)
tokenizer.save_pretrained(output_name)


