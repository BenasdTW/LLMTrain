from datasets import load_dataset

wikisql = load_dataset("wikisql")

# Access the training set
train_data = wikisql['train']

# Get the first example
first_example = train_data[0]
print(first_example['question'])  # Natural language question
print(first_example['sql'])       # SQL query
print(first_example['table'])     # Table data
