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

    # ==========================================================
    # CREATE PREDICTIONS TABLE
    # ==========================================================

    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        tweet_text TEXT NOT NULL,
        sentiment VARCHAR(20) NOT NULL,
        actual_sentiment VARCHAR(20),
        confidence DECIMAL(5,2),
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )
    """

    cursor.execute(create_table_query)


    # ==========================================================
    # ADD actual_sentiment TO EXISTING TABLE
    # ==========================================================
    # This is important because your PostgreSQL table may
    # already exist from the previous version.

    alter_table_query = """
    ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS actual_sentiment VARCHAR(20)
    """

    cursor.execute(alter_table_query)


    # ==========================================================
    # CHANGE created_at TO TIMESTAMPTZ
    # ==========================================================
    # This keeps the timestamp with timezone information.

    alter_timestamp_query = """
    ALTER TABLE predictions
    ALTER COLUMN created_at TYPE TIMESTAMPTZ
    USING created_at AT TIME ZONE 'UTC'
    """

    cursor.execute(alter_timestamp_query)


    conn.commit()

    cursor.close()
    conn.close()

    print("PostgreSQL table 'predictions' is ready!")


if __name__ == "__main__":
    setup_database()