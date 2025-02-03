import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
import bitsandbytes as bnb
from trl import SFTTrainer, SFTConfig
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from configs_and_helpers import clear_memory, vl_format_data, generate_text_from_sample

apply_liger_kernel_to_qwen2_vl()

output_name = "qwen2vl_test"
model_id = "Qwen/Qwen2-VL-2B-Instruct"
dataset_id = "HuggingFaceM4/ChartQA"

system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=["train[:2%]", "val[:2%]", "test[:1%]"])

train_dataset = [vl_format_data(sample, system_message) for sample in train_dataset]
eval_dataset = [vl_format_data(sample, system_message) for sample in eval_dataset]
test_dataset = [vl_format_data(sample, system_message) for sample in test_dataset]


model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)

processor = Qwen2VLProcessor.from_pretrained(model_id)
print(f"{train_dataset[0]=}")
print(f"{train_dataset[0][1:2]=}")


output = generate_text_from_sample(model, processor, train_dataset[0])
print(f"{output=}")

clear_memory(globals())

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
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
processor.padding_side = "right"  # Ensure padding is added to the right side
processor.tokenizer.padding_side = "right"  # Ensure padding is added to the right side

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
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    eval_accumulation_steps=4,
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
    dataset_kwargs={"skip_prepare_dataset": True},
    # max_seq_length=1024  # Maximum sequence length for input
    remove_unused_columns = False  # Keep unused columns in dataset
)

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
    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    batch["labels"] = labels

    return batch

print(f"Processed data:\n{collator_fn(train_dataset[:2])}")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator_fn,
    peft_config=peft_config,
    processing_class=processor.tokenizer,
    optimizers=(optimizer, scheduler)
)
trainer.train()
trainer.save_model(training_args.output_dir)

# model.save_model(output_name)
