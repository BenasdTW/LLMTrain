import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from liger_kernel.transformers import AutoLigerKernelForCausalLM
from datasets import load_dataset
import torch

# Load the WikiSQL dataset and the fine-tuned model
# model_name = "meta-llama/Llama-3.2-1B"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "./finetuned-llama-text2sql"
# model_name = "./new-text2sql"
# model_name = "./text2sql-1b"
model_name = "./text2sql-1b-Instruct-format"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
# Load the Base Model
# model = AutoLigerKernelForCausalLM.from_pretrained(
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    # quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,  # Match input type
)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)  # Match input type
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# model.generate automatically add a BOS token to the front of the string, therefore it's no point adding it manually here.
model_template = """<|start_header_id|>system<|end_header_id|>

{schema}

-- {system}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""
def split_string(input_string):
    # Split using '--', allowing optional newlines (\n) before/after
    parts = re.split(r"\s*\n?\s*--\s*\n?\s*", input_string.strip())
    # Filter out empty strings
    return [part.strip() for part in parts if part.strip()]


# text = """CREATE TABLE test_data (
#     id INT PRIMARY KEY,
#     product VARCHAR(10),
#     location VARCHAR(10),
#     test_code_num INT,
#     test_code_en VARCHAR(50)
# );

# -- Using valid SQLite, answer the following questions for the tables provided above.

# -- which product has most test_code_num=1"""
# -- how many stadiums in total?
text = """CREATE TABLE stadium (
    stadium_id number,
    location text,
    name text,
    capacity number,
)

-- Using valid SQLite, answer the following questions for the tables provided above.

-- Which city has the highest average stadium capacity?"""

# Split the string into 3 parts and format it with the model template
split = split_string(text)
new_text = model_template.format(schema=split[0], system=split[1], question=split[2])

print(f"{new_text=}")

model_input = tokenizer(new_text, return_tensors="pt").to("cuda")
print(model_input)

generated_ids = model.generate(**model_input, max_new_tokens=100, num_beams=3, early_stopping=True)
            
print("Result")
print(tokenizer.decode(generated_ids[0], skip_special_tokens=False))
