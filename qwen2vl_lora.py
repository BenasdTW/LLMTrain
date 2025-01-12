import torch
import gc
import time
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
# from liger_kernel.transformers import AutoLigerKernelForCausalLM
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
import bitsandbytes as bnb
from trl import SFTTrainer, SFTConfig
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

apply_liger_kernel_to_qwen2_vl()

model_id = "Qwen/Qwen2-VL-2B-Instruct"
dataset_id = "HuggingFaceM4/ChartQA"

system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

def format_data(sample):
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

train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=["train[:1%]", "val[:1%]", "test[:1%]"])

train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
test_dataset = [format_data(sample) for sample in test_dataset]


model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)

processor = Qwen2VLProcessor.from_pretrained(model_id)
print(f"{train_dataset[0]=}")
print(f"{train_dataset[0][1:2]=}")


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
    ).to(
        device
    )  # Move inputs to the specified device
    print(f"{model_inputs=}")

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text

output = generate_text_from_sample(model, processor, train_dataset[0])
print(f"{output=}")

def clear_memory():
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


clear_memory()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    use_cache=False
)
processor = Qwen2VLProcessor.from_pretrained(model_id)

# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# Apply PEFT model adaptation
print(model)
model = get_peft_model(model, peft_config)
print(model)

# Print trainable parameters
model.print_trainable_parameters()

optimizer = create_loraplus_optimizer(
    model=model,
    optimizer_cls=bnb.optim.PagedAdamW8bit,
    # optimizer_cls=torch.optim.AdamW,
    lr=2e-4,
    eps=1e-6,
    # eps=1e-8,
    betas=(0.9, 0.999),
    weight_decay=0.0,
    loraplus_lr_ratio=16,
)
scheduler = None


# Configure training arguments
training_args = SFTConfig(
    output_dir="qwen2-2b-instruct-trl-sft-ChartQA",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    eval_accumulation_steps=8,
    # Logging and evaluation
    logging_steps=1,
    eval_steps=10,
    torch_empty_cache_steps=1,
    eval_strategy="steps",
    save_strategy="epoch",
    bf16=True,
    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={"use_reentrant": False},
    gradient_checkpointing=True,
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    # max_seq_length=1024  # Maximum sequence length for input
    remove_unused_columns = False  # Keep unused columns in dataset
)

# training_args.remove_unused_columns = False  # Keep unused columns in dataset
# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    batch["labels"] = labels  # Add labels to the batch

    return batch

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
    optimizers=(optimizer, scheduler)
)
trainer.train()
trainer.save_model(training_args.output_dir)
# clear_memory()
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_id,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
# )

# processor = Qwen2VLProcessor.from_pretrained(model_id)