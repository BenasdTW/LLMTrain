from ollama import Client
from sql_test import query

db_config = {
    "host": "140.118.152.230",
    "user": "root",
    "password": "test",
    "database": "test_db",
}

client = Client(host="http://172.17.0.1:11434/")
model = "duckdb-nsql"

sys_prompt = """Here is the database schema that the SQL query will run on:
CREATE TABLE test_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product VARCHAR(10) NOT NULL,
    location VARCHAR(10) NOT NULL,
    test_code_num INT NOT NULL,
    test_code_en VARCHAR(50) NOT NULL
);"""

# user_prompt = "which product has most test_code_num=1"
# SELECT product FROM test_data WHERE test_code_num = 1;
# user_prompt = "Which product has the highest count of test_code_num = 1?"
# SELECT product FROM test_data WHERE test_code_num = 1 GROUP BY product ORDER BY COUNT(*) DESC LIMIT 1;
# user_prompt = "哪個產品的有最多 test_code_num = 1"
# SELECT product FROM test_data WHERE test_code_num = 1;
user_prompt = "哪個產品的 test_code_num = 1 數量最多"
# SELECT product FROM test_data WHERE test_code_num = 1 GROUP BY product ORDER BY COUNT(*) DESC LIMIT 1;

r = client.generate(
    model=model,
    system=sys_prompt,
    prompt=user_prompt
)

print("System prompt:")
print(sys_prompt)
print()

print("User prompt:")
print(user_prompt)
print()

print("Generated query:")
print(r["response"])
print()

print("Query result:")
print(query(db_config, r["response"]))
print()

