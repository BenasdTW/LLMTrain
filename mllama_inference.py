import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from lora_config import quantization_config, lora_config
from liger_kernel.transformers import apply_liger_kernel_to_mllama
apply_liger_kernel_to_mllama()

model_id = "meta-llama/Llama-3.2-11B-Vision"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

print(model)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "<|image|><|begin_of_text|>If I had to write a haiku for this one"
inputs = processor(image, prompt, return_tensors="pt").to(model.device)
print(inputs)

model.eval()
output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))
