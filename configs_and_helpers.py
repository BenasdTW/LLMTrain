import time
import gc
import torch
import bitsandbytes as bnb
from peft import LoraConfig
from peft.optimizers import create_loraplus_optimizer
from transformers.trainer_utils import IntervalStrategy
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTConfig
from qwen_vl_utils import process_vision_info


def vl_format_data(sample, system_message=""):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample["query"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"][0]}],
        },
    ]

def clear_memory(scope):
    # Delete variables if they exist in the current global scope
    if "inputs" in scope:
        del scope["inputs"]
    if "model" in scope:
        del scope["model"]
    if "processor" in scope:
        del scope["processor"]
    if "trainer" in scope:
        del scope["trainer"]
    if "peft_model" in scope:
        del scope["peft_model"]
    if "bnb_config" in scope:
        del scope["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[1:2], tokenize=False, add_generation_prompt=True  # Use the sample without the system message
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample)
    print(f"{image_inputs=}")

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)
    print(f"{model_inputs=}")

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Configure QLoRA with PEFT
def lora_config_builder(r: int = 32, lora_alpha: int = 16, target_modules: str | list = "all-linear", modules_to_save: list = ["input_layernorm", "post_attention_layernorm", "norm"], lora_dropout: float = 0.05):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,  # Target attention layers
        modules_to_save=modules_to_save,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return lora_config



# 1B: 1:55:43, 3B: 4:59:29, 8B: 10:29:01
# Define Training Arguments
# Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
#     = 256
def training_args_builder(output_name: str, eff_batch: int = 256, device_batch: int = 8, gpu_count: int = 1, eval_batch = None, eval_accumulation_steps = 1, lr = 2e-4, epochs = 3):
    if eval_batch is None:
        eval_batch = device_batch
    training_args = SFTConfig(
        output_dir="./output",
        per_device_train_batch_size=device_batch,
        gradient_accumulation_steps=(eff_batch // device_batch) // gpu_count,
        per_device_eval_batch_size=eval_batch,
        eval_accumulation_steps=eval_accumulation_steps,
        # torch_empty_cache_steps=1,
        learning_rate=lr,
        num_train_epochs=epochs,
        use_liger=True,
        logging_dir=f"./profile/{output_name}",
        logging_steps=1,
        report_to="tensorboard",
        save_strategy=IntervalStrategy.EPOCH,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=8,
        eval_on_start=True,
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return training_args

def loraplus_optimizer_builder(model, bits=32, lr=2e-4, eps=1e-8, betas=(0.9, 0.999), weight_decay=0.0, loraplus_lr_ratio=16):
    if bits != 32:
        optim = bnb.optim.PagedAdamW8bit
    else:
        optim = torch.optim.AdamW
    optimizer = create_loraplus_optimizer(
        model=model,
        optimizer_cls=optim,
        lr=lr,
        eps=eps,
        betas=betas,
        weight_decay=weight_decay,
        loraplus_lr_ratio=loraplus_lr_ratio,
    )
    return optimizer, None
