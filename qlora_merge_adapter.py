from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

model_original = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "./text2sql-3b-Instruct-loraplus-extra"

model_original = "Qwen/Qwen2-VL-2B-Instruct"
model_name = "./qwen2-2b-instruct-trl-sft-ChartQA"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
# Load the Base Model
# model = AutoLigerKernelForCausalLM.from_pretrained(
# model = Qwen2VLForConditionalGeneration.from_pretrained(
model = AutoModelForCausalLM.from_pretrained(
    # model_name, 
    model_original, 
    device_map="auto", 
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,  # Match input type
)
print("=" * 60)
print("Load original model in 4bits")
print(model)
model = model.dequantize()

print("=" * 60)
print("Dequantize the model to bf16")
print(model)
model = PeftModel.from_pretrained(model = model, model_id = model_name)

print("=" * 60)
print("Load unmerged peft adapter")
print(model)
model = model.merge_and_unload()

print("=" * 60)
print("Merge peft model")
print(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)

output_name = "merged"
model.save_pretrained(output_name)
tokenizer.save_pretrained(output_name)
print(f"Model is saved as {output_name}")
