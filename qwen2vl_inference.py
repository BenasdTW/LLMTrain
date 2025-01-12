import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# model_name = "Qwen/Qwen2-VL-2B-Instruct"
model_name = "Qwen/Qwen2-VL-2B-Instruct-AWQ"
# model_name = "unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(model_name)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            # {"type": "text", "text": "Descript the image. Show me the text in the image"},
            # {"type": "text", "text": "轉錄圖中的所有文字及符號，請以圖片內原本的語言呈現，不要翻譯，也不要多做解釋:"},
            # {"type": "text", "text": "Transcibe the image to text. Do not translate it, and do not explain anything."},
            # {"type": "text", "text": "Transcibe the image to text without adding explaination. Do not translate it."},
            {"type": "text", "text": "Transcibe the image without adding explaination. Use Traditional Chinese instead of Simplfied Chinese."},
            # {"type": "text", "text": "Recognize all the text in the image without adding explaination."},
            # {"type": "text", "text": "Show me the text on the image"},
            {
                "type": "image",
                # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                "image": "file://./test.png",
            },
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
)
print(inputs)
inputs = inputs.to("cuda")

# Inference: Generation of the output
print(model)
model.eval()
# generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)