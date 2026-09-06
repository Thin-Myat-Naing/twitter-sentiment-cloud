import os
import psycopg2


def get_db_connection():
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        raise Exception("DATABASE_URL is not set.")

    return psycopg2.connect(database_url)


def setup_database():
    conn = get_db_connection()
    cursor = conn.cursor()

    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        tweet_text TEXT NOT NULL,
        sentiment VARCHAR(20) NOT NULL,
        confidence DECIMAL(5,2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    cursor.execute(create_table_query)
    conn.commit()

    cursor.close()
    conn.close()

    print("PostgreSQL table 'predictions' created successfully!")


if __name__ == "__main__":
    setup_database()