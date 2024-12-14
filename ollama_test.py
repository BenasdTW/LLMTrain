from ollama import Client
import json

conversation = [
    {
        "role": "system",
        "content": "You are a bored assistant. Provide short answers."
    },
    {
        "role": "user",
        "content": "Why is the sky blue?"
    },
    {
        "role": "user",
        "content": "Because the gods wanted it that way."
    },
    {
        "role": "user",
        "content": "Why did the gods want it that way?"
    },
]

model_template = """<|start_header_id|>system<|end_header_id|>

You are a text-to-SQL model. Please generate a SQL query command according to the user input. Only generates the SQL query, do not explain anything.
This is the SQL schema:
{{ .Schema }}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>


{{ .TextQuery }}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{{ .SQLQuery }}
"""

ollama = Client(host="http://172.17.0.3:11434")
response = ollama.chat(
    model="llama3.2:1b",
    messages=conversation,
    options={"temperature": 0}
)
print(response["message"]["content"])

response = ollama.generate(
    model="llama3.2:1b",
    prompt=json.dumps(conversation),
    options={"temperature": 0}
)
print(response["response"])
