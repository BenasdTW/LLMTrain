import re
import torch
import json
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import TrainerCallback

# model_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# {schema}

# -- {system}
# <|eot_id|>
# <|start_header_id|>user<|end_header_id|>

# {question}<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>

# {query}<|end_of_text|>"""

model_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{schema}

-- {system}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""
# output_template = "{query}<|end_of_text|>"

class NSText2SQLDatasetFormatted(Dataset):
    def __init__(self, tokenizer, size=None, max_length=2048, split="train"):
        self.tokenizer = tokenizer

        dataset = load_dataset("NumbersStation/NSText2SQL", split=split)
        self.dataset = dataset.filter(lambda x: self._filter_by_token_count(x, max_length))
        print(f"{len(self.dataset)=}")

        if size:
            self.dataset = self.dataset.select(range(size))
        self.max_length = max_length

        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        model_inputs, instruction_len = self._tokenize(self.dataset[index]["instruction"], self.dataset[index]["output"], self.max_length)

        model_inputs["input_ids"] = model_inputs["input_ids"].squeeze(0)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0)
        labels = model_inputs["input_ids"].clone()  # Labels for causal LM
        # Mask out the instruction string
        labels[:instruction_len] = -100
        # Mask out the paddings
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
            
        return model_inputs

    def _split_string(self, input_string):
        # Split using '--', allowing optional newlines (\n) before/after
        parts = re.split(r"\s*\n?\s*--\s*\n?\s*", input_string.strip())
        # Filter out empty strings
        return [part.strip() for part in parts if part.strip()]

    def _tokenize(self, inst, output, max_length):
        split = self._split_string(inst)
        if len(split) != 3:
            # 26 fails
            return None, None
        input_str = model_template.format(schema=split[0], system=split[1], question=split[2])
        instruction_len = len(torch.tensor(self.tokenizer.encode(input_str), dtype=torch.int32))

        input_str = input_str + output + self.tokenizer.eos_token
        model_inputs = self.tokenizer(input_str, max_length=max_length, padding="max_length", return_tensors="pt")
        return model_inputs, instruction_len
        
    def _filter_by_token_count(self, x, max_length):
        # Tokenize the text and check the length
        model_inputs, _ = self._tokenize(x["instruction"], x["output"], max_length)
        if model_inputs is None:
            return False
        return len(model_inputs["input_ids"].squeeze(0)) <= max_length

class NSText2SQLDataset(Dataset):
    def __init__(self, tokenizer, size=None, max_length=2048, split="train"):
        self.tokenizer = tokenizer

        dataset = load_dataset("NumbersStation/NSText2SQL", split=split)
        self.dataset = dataset.filter(lambda x: self._filter_by_token_count(x, max_length))
        print(f"{len(self.dataset)=}")

        if size:
            self.dataset = self.dataset.select(range(size))
        self.max_length = max_length

        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        model_inputs, instruction_len = self._tokenize(self.dataset[index]["instruction"], self.dataset[index]["output"], self.max_length)

        model_inputs["input_ids"] = model_inputs["input_ids"].squeeze(0)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0)
        labels = model_inputs["input_ids"].clone()  # Labels for causal LM
        # Mask out the instruction string
        labels[:instruction_len] = -100
        # Mask out the paddings
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
            
        return model_inputs

    def _tokenize(self, input_str, output, max_length):
        instruction_len = len(torch.tensor(self.tokenizer.encode(input_str), dtype=torch.int32))

        input_str = input_str + output + self.tokenizer.eos_token
        model_inputs = self.tokenizer(input_str, max_length=max_length, padding="max_length", return_tensors="pt")
        return model_inputs, instruction_len
        
    def _filter_by_token_count(self, x, max_length):
        # Tokenize the text and check the length
        model_inputs, _ = self._tokenize(x["instruction"], x["output"], max_length)
        if model_inputs is None:
            return False
        return len(model_inputs["input_ids"].squeeze(0)) <= max_length

class CustomLoggingCallback(TrainerCallback):
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name
    def on_train_begin(self, args, state, control, **kwargs):
        with open(self.file_name, "w") as f:
            pass
        print(f"Started")
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["step"] = state.global_step
            with open(self.file_name, "a") as f:
                obj_str = json.dumps(logs)
                f.write(obj_str)
                f.write("\n")


# max_length = 512: 220742
# dataset = NSText2SQLDataset(size=102500, max_length=512)
# dataset, eval_set = torch.utils.data.random_split(dataset, [102400, 100])
# print(f"{len(dataset)=} {len(eval_set)=}")


