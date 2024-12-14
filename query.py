import sqlite3

def query(db_name, select_query):
    # Connect to the MySQL database
    try:
        connection = sqlite3.connect(db_name)
        cursor = connection.cursor()

        cursor.execute(select_query)

        # Fetch all rows from the query result
        rows = cursor.fetchall()

        return rows

    finally:
        if connection:
            cursor.close()
            connection.close()


def main():
    # select_query = "SELECT product FROM test_data WHERE test_code_num = 1;"
    select_query = "SELECT product FROM test_data WHERE test_code_num = 1 GROUP BY product ORDER BY COUNT(*) DESC LIMIT 1;"
    query("db/test.db", select_query)

