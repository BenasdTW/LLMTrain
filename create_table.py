import sqlite3

conn = sqlite3.connect("db/test.db")
print("Connected")

# Create the table
create_table_query = """
CREATE TABLE IF NOT EXISTS test_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product VARCHAR(10) NOT NULL,
    location VARCHAR(10) NOT NULL,
    test_code_num INT NOT NULL,
    test_code_en VARCHAR(50) NOT NULL
);
"""

# Data to insert
data = {
    'product': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'location': ['L1', 'L1', 'L2', 'L2', 'L1', 'L2', 'L2', 'L1', 'L1', 'L2'],
    'test_code_num': [0, 1, 1, 1, 0, 1, 2, 0, 2, 2],
    'test_code_en': ['no_signal', 'signal', 'signal', 'signal', 'no_signal', 'signal', 'error', 'no_signal', 'error', 'error']
}

insert_query = """
INSERT INTO test_data (product, location, test_code_num, test_code_en)
VALUES (?, ?, ?, ?);
"""

try:
    cursor = conn.cursor()
    cursor.execute(create_table_query)

    # Insert data into the table
    rows = zip(data['product'], data['location'], data['test_code_num'], data['test_code_en'])
    cursor.executemany(insert_query, rows)

    # Commit changes
    conn.commit()
    print("Table created and data inserted successfully!")

finally:
    cursor.close()
    conn.close()

