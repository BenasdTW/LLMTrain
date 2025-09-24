import torch
from datasets import Dataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
# from unsloth import FastLanguageModel 
from trl import SFTConfig, SFTTrainer

model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

model = AutoModelForCausalLM.from_pretrained(model_id)
# model = AutoLigerKernelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = model_id,
#     max_seq_length = 16,
#     dtype = torch.bfloat16,
#     load_in_4bit = False,
# )
# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",],
#     lora_alpha = 16,
#     lora_dropout = 0, # Supports any, but = 0 is optimized
#     bias = "none",    # Supports any, but = "none" is optimized
#     # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
#     use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
#     random_state = 3407,
#     use_rslora = False,  # We support rank stabilized LoRA
#     loftq_config = None, # And LoftQ
# )
tokenizer.pad_token = tokenizer.eos_token

dummy_dataset = Dataset.from_dict({"text": ["Dummy dataset"] * 16, })

training_args = SFTConfig(
    num_train_epochs=10,
    per_device_train_batch_size=4,
    report_to="none",
    # use_liger=True,
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dummy_dataset,
    processing_class=tokenizer,
)

trainer.train()
