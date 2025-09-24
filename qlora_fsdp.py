# accelerate launch --config_file fsdp.yaml t3.py
import torch
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset
from transformers.trainer_utils import IntervalStrategy
from accelerate import PartialState

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.bos_token
tokenizer.pad_token_id = 151643

dataset = Dataset.from_dict({"text": ["Hello world"] * 16, })
eval_set = Dataset.from_dict({"text": ["Hello world"] * 16, })

bnb_conf = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_quant_storage=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    use_cache=False,
    device_map={"": PartialState().process_index},
    attn_implementation="flash_attention_2",
    # quantization_config=bnb_conf,
    torch_dtype=torch.bfloat16,
)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Configure LoRA adapters
model = get_peft_model(model, lora_config)

training_args = SFTConfig(
    output_dir="./output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=2,
    eval_accumulation_steps=1,
    logging_steps=1,
    bf16=True,
    gradient_checkpointing=True,
    save_strategy=IntervalStrategy.EPOCH,
    # gradient_checkpointing_kwargs={"use_reentrant": False}
    gradient_checkpointing_kwargs={"use_reentrant": True}
)
model.enable_input_require_grads()

trainer = SFTTrainer(
    model=model,
    # peft_config=lora_config,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_set,
)

# Start training
trainer.model.print_trainable_parameters()
trainer.train()

if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model()
