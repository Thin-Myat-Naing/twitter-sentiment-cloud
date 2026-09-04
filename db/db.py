import mysql.connector


def setup_database():

    # ==========================================================
    # 1. Connect to MySQL without selecting a database
    # ==========================================================

    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root"
    )

    mycursor = mydb.cursor()

    # Database name
    db_name = "`Cloud-Twitter Sentiment`"

    # Create database if it does not exist
    mycursor.execute(
        f"CREATE DATABASE IF NOT EXISTS {db_name}"
    )

    print("Database created or verified successfully!")

    mycursor.close()
    mydb.close()


    # ==========================================================
    # 2. Connect to the database
    # ==========================================================

    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="Cloud-Twitter Sentiment"
    )

    mycursor = mydb.cursor()


    # ==========================================================
    # 3. Create predictions table
    # ==========================================================

    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        tweet_text TEXT NOT NULL,
        sentiment VARCHAR(20) NOT NULL,
        confidence DECIMAL(5,2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    mycursor.execute(create_table_query)

    mydb.commit()

    print("Table 'predictions' created successfully!")


    # ==========================================================
    # 4. Close connection
    # ==========================================================

    mycursor.close()
    mydb.close()


if __name__ == "__main__":
    setup_database()