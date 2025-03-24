import torch
from datasets import Dataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from accelerate import PartialState
from peft import get_peft_model, LoraConfig

# model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
model_id = "Qwen/Qwen2.5-1.5B-Instruct"


bnb_conf = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    # For FSDP
    bnb_4bit_quant_storage=torch.bfloat16,
)

# model = AutoModelForCausalLM.from_pretrained(
model = AutoLigerKernelForCausalLM.from_pretrained(
    model_id,
    device_map={"": PartialState().process_index},
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,  # Match input type
    use_cache=False,
    quantization_config=bnb_conf,  # QLoRA (NF4, DQ)
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.bos_token
# tokenizer.pad_token_id = 151643
print(f"{tokenizer.eos_token_id=}")
print(f"{tokenizer.pad_token_id=}")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

dummy_dataset = Dataset.from_dict({"text": ["Dummy dataset"] * 32, })

training_args = SFTConfig(
    num_train_epochs=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=2,
    eval_accumulation_steps=1,
    learning_rate=5e-5,
    report_to="none",
    use_liger_kernel=True,
    bf16=True,
    eval_on_start=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dummy_dataset,
    eval_dataset=dummy_dataset,
    processing_class=tokenizer,
)

# trainer.train()
trainer.model.print_trainable_parameters()
trainer.train()

# Save the LoRA adapter model
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model()
