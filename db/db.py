import os
import psycopg2


def setup_database():
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        print("DATABASE_URL is not set.")
        return

    try:
        conn = psycopg2.connect(database_url)
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

        print("PostgreSQL table 'predictions' created successfully!")

        cursor.close()
        conn.close()

    except Exception as e:
        print("Database setup error:", e)


if __name__ == "__main__":
    setup_database()