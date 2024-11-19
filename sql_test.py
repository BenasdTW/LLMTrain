import pymysql

db_config = {
    'host': '192.168.1.109',
    'user': 'root',
    'password': 'test',
    'database': 'test_db',
}

# Connect to the MySQL database
try:
    connection = pymysql.connect(**db_config)
    cursor = connection.cursor()

    # select_query = "SELECT product FROM test_data WHERE test_code_num = 1;"
    select_query = "SELECT product FROM test_data WHERE test_code_num = 1 GROUP BY product ORDER BY COUNT(*) DESC LIMIT 1;"
    cursor.execute(select_query)

    # Fetch all rows from the query result
    rows = cursor.fetchall()

    print("-" * 60)
    print(rows)

except pymysql.MySQLError as e:
    print(f"Error: {e}")

finally:
    if connection:
        cursor.close()
        connection.close()
