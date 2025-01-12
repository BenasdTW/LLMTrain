import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from lora_config import quantization_config, lora_config
from liger_kernel.transformers import apply_liger_kernel_to_mllama
apply_liger_kernel_to_mllama()

# model_id = "meta-llama/Llama-3.2-11B-Vision"
model_id = "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    # quantization_config=quantization_config,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

print(model)

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image_path = "test.png"
image = Image.open(image_path)
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        # {"type": "text", "text": "轉錄圖中的所有文字及符號，請以圖片內原本的語言呈現，不要翻譯，也不要多做解釋:"}
        {"type": "text", "text": "Transcibe the image without adding explaination. Use Traditional Chinese instead of Simplfied Chinese."},
        # {"type": "text", "text": "Transcribe the image to text. Do not translate it, and do not explain anything."}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device)

# prompt = "<|image|><|begin_of_text|>轉錄圖中的所有文字及符號:\n"
# inputs = processor(image, prompt, return_tensors="pt").to(model.device)
print(inputs)

model.eval()
output = model.generate(**inputs, max_new_tokens=300)
print(processor.decode(output[0]))
