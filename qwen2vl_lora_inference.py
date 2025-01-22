import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from configs_and_helpers import vl_format_data, generate_text_from_sample

apply_liger_kernel_to_qwen2_vl()

# model_id = "Qwen/Qwen2-VL-2B-Instruct"
model_id = "merged"
# model_id = "./qwen2-2b-instruct-trl-sft-ChartQA"
# adapter_name = "./qwen2-2b-instruct-trl-sft-ChartQA"
dataset_id = "HuggingFaceM4/ChartQA"

system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=["train[:1%]", "val[:2%]", "test[:1%]"])

train_dataset = [vl_format_data(sample, system_message) for sample in train_dataset]

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
# model.load_adapter(adapter_name)

# model.merge_and_unload()

processor = Qwen2VLProcessor.from_pretrained(model_id)
print(f"{train_dataset[0]=}")
print(f"{train_dataset[0][1:2]=}")


output = generate_text_from_sample(model, processor, train_dataset[0])
print(f"{output=}")

# clear_memory(globals())

