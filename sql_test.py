import pymysql

def query(db_config, select_query):
    
    # Connect to the MySQL database
    try:
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()

        cursor.execute(select_query)

        # Fetch all rows from the query result
        rows = cursor.fetchall()

        return rows

    except pymysql.MySQLError as e:
        print(f"Error: {e}")

    finally:
        if connection:
            cursor.close()
            connection.close()

db_config = {
    "host": "140.118.152.230",
    "user": "root",
    "password": "test",
    "database": "test_db",
}

# select_query = "SELECT product FROM test_data WHERE test_code_num = 1;"
select_query = "SELECT product FROM test_data WHERE test_code_num = 1 GROUP BY product ORDER BY COUNT(*) DESC LIMIT 1;"
query(db_config, select_query)
