import torch
from datasets import Dataset
from trl import GKDConfig, GKDTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers.trainer_utils import IntervalStrategy
from configs_and_helpers import quantization_config, lora_config
from peft import get_peft_model
from peft import LoraConfig
from peft.optimizers import create_loraplus_optimizer

NUM_DUMMY_SAMPLES = 100

output_name = "distiled"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# The model to optimize
model = AutoLigerKernelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device_map="auto", 
    attn_implementation="flash_attention_2",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,  # Match input type
)
# The teacher model to calculate the KL divergence against
teacher_model = AutoLigerKernelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device_map="auto", 
    attn_implementation="flash_attention_2",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,  # Match input type
)

# Configure QLoRA with PEFT
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention layers
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Target attention layers
    # modules_to_save=["input_layernorm", "post_attention_layernorm", "norm"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# model = get_peft_model(model, lora_config)

# optimizer = create_loraplus_optimizer(
#     model=model,
#     optimizer_cls=bnb.optim.PagedAdamW8bit,
#     # optimizer_cls=torch.optim.AdamW,
#     lr=5e-5,
#     eps=1e-6,
#     # eps=1e-8,
#     betas=(0.9, 0.999),
#     weight_decay=0.0,
#     loraplus_lr_ratio=16,
# )
# scheduler = None

train_dataset = Dataset.from_dict(
    {
        "messages": [
            [
                {"role": "user", "content": "Hi, how are you?"},
                {"role": "assistant", "content": "I'm great thanks"},
            ]
        ]
        * NUM_DUMMY_SAMPLES
    }
)
eval_dataset = Dataset.from_dict(
    {
        "messages": [
            [
                {"role": "user", "content": "What colour is the sky?"},
                {"role": "assistant", "content": "The sky is blue"},
            ]
        ]
        * NUM_DUMMY_SAMPLES
    }
)

training_args = GKDConfig(
    output_dir="test",
    per_device_train_batch_size=6,
    logging_steps=1,
    eval_strategy=IntervalStrategy.STEPS,
    eval_steps=8,
    # use_liger=True,
    logging_dir=f"./profile/{output_name}",
    logging_steps=1,
    report_to="tensorboard",
    use_liger_kernel=True,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
trainer = GKDTrainer(
    model=model,
    peft_config=lora_config,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()