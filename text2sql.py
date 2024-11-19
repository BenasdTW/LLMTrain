from ollama import Client

client = Client(host="http://ai-twins.co:10138/")
model = "duckdb-nsql:7b-q8_0"
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

print(r['response'])