# accelerate launch finetune_example_fsdp.py
import re
import torch
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from configs_and_helpers import quantization_config, lora_config_builder, loraplus_optimizer_builder, training_args_builder
from datasets import load_dataset
from accelerate import PartialState

output_name = "qwen25_qlora_32_8"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# model_name = "Qwen/Qwen2.5-32B-Instruct"
dataset_name = "NumbersStation/NSText2SQL"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the pad token to the bos token (151643), make sure to check the tokenizer_config.json of the base model
tokenizer.pad_token = tokenizer.bos_token
tokenizer.pad_token_id = 151643
print(f"{tokenizer.eos_token_id=}")
print(f"{tokenizer.pad_token_id=}")

# max_length = 512: 220742
max_length = 512

def split_string(input_string):
    # Split using '--', allowing optional newlines (\n) before/after
    parts = re.split(r"\s*\n?\s*--\s*\n?\s*", input_string.strip())
    # Filter out empty strings
    return [part.strip() for part in parts if part.strip()]

def format_and_tokenize(expample):
    split = split_string(expample["instruction"])
    # Filter out examples that don't follow the format
    if len(split) != 3:
        # 26 examples are not in the correct format
        return {"input_ids": None, "attention_mask": None, "labels": None}
    sys_prompt = f"{split[0]}\n\n-- {split[1]}"
    user_prompt = split[2]

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")

    input_str = input_str + expample["output"] + tokenizer.eos_token
    model_inputs = tokenizer(input_str, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")

    input_ids = model_inputs["input_ids"].squeeze(0)
    # assert len(input_ids) == max_length
    labels = input_ids.clone()
    # Label masking
    # Replace padding token id with -100 in labels
    labels[labels == tokenizer.pad_token_id] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": model_inputs["attention_mask"].squeeze(0),
        "labels": labels,
    }

    return out

# Filter out examples that don't follow the format
def filter_func(example):
    if example["input_ids"] is None:
        return False

    return True if example["labels"][-1] == -100 else False

# Preprocessing
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.map(format_and_tokenize, num_proc=16)

used_columns = ["input_ids", "attention_mask", "labels"]
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in used_columns])
dataset = dataset.filter(filter_func, num_proc=16)
dataset = dataset.select(range(21504))
splits = dataset.train_test_split(test_size=1024)
dataset = splits["train"]
eval_set = splits["test"]

print(f"{dataset=}")

bnb_conf = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    # For FSDP
    bnb_4bit_quant_storage=torch.bfloat16,
)
# Load the base model (use liger kernel)
model = AutoLigerKernelForCausalLM.from_pretrained(
# model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map={"": PartialState().process_index},
    use_cache=False,
    attn_implementation="flash_attention_2",
    # quantization_config=bnb_conf,  # QLoRA (NF4, DQ)
    torch_dtype=torch.bfloat16,  # Match input type
)

# Configure LoRA adapters
model = get_peft_model(model, lora_config_builder())

training_args = training_args_builder(output_name, eff_batch=128, device_batch=32, gpu_count=2, eval_batch=16, eval_accumulation_steps=2, epochs=2)
# training_args.ddp_find_unused_parameters = False
# training_args.gradient_checkpointing_kwargs={"use_reentrant": True}
# model.enable_input_require_grads()
class MySFTTrainer(SFTTrainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        # LoRA+ Optimizer (ratio = 8)
        self.optimizer, _ = loraplus_optimizer_builder(self.model, lr=2e-4, loraplus_lr_ratio=8)
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

# trainer = MySFTTrainer(
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_set,
    # optimizers=optim,
)
# print(f"{trainer.optimizer=}")

# Start training
trainer.model.print_trainable_parameters()
trainer.train()

# Save the LoRA adapter model
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model()

