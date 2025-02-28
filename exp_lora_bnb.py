# CUDA_VISIBLE_DEVICES=1 /opt/conda/bin/python /workspaces/LLMTrain/exp_lora.py
import torch
from trl import SFTTrainer
from peft import get_peft_model
from configs_and_helpers import quantization_config, lora_config_builder, loraplus_optimizer_builder, training_args_builder, vl_format_data
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
from peft import get_peft_model
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLProcessor

apply_liger_kernel_to_qwen2_vl()

output_name = "qwen2vl_lora_bnb"
model_id = "Qwen/Qwen2-VL-2B-Instruct"
dataset_name = "HuggingFaceM4/ChartQA"
system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

train_dataset, eval_dataset, test_dataset = load_dataset(dataset_name, split=["train[:20%]", "val[:4%]", "test[:1%]"])

train_dataset = [vl_format_data(sample, system_message) for sample in train_dataset]
eval_dataset = [vl_format_data(sample, system_message) for sample in eval_dataset]
test_dataset = [vl_format_data(sample, system_message) for sample in test_dataset]

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    # device_map={"": accelerator.process_index},
    # device_map="cuda:0", 
    # torch_dtype=torch.float16,
    torch_dtype=torch.bfloat16,  # Match input type
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2",
    use_cache=False
)

processor = Qwen2_5_VLProcessor.from_pretrained(model_id)
processor.padding_side = "right"  # Ensure padding is added to the right side
processor.tokenizer.padding_side = "right"  # Ensure padding is added to the right side

print(f"{processor.tokenizer.eos_token_id=}")
print(f"{processor.tokenizer.pad_token_id=}")

# Configure LoRA adapters
model = get_peft_model(model, lora_config_builder())

# LoRA+ Optimizer (ratio = 16)
optim = loraplus_optimizer_builder(model, lr=2e-4)

training_args = training_args_builder(output_name, eff_batch=128, device_batch=8, epochs=3)
training_args.dataset_kwargs = {"skip_prepare_dataset": True}

# Create a data collator to encode text and image pairs
def collator_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    # Mask padding tokens in labels
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2_5_VLProcessor):
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    batch["labels"] = labels

    return batch

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator_fn,
    processing_class=processor.tokenizer,
    optimizers=optim,
)

# Start training
trainer.model.print_trainable_parameters()
trainer.train()

# Save the LoRA adapter model
trainer.save_model(training_args.output_dir)

# 11.021 GB
# 1:48:48
