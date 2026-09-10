from flask import Flask, render_template, request, jsonify
import joblib
import os
import psycopg2
from psycopg2.extras import RealDictCursor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================================
# FLASK APPLICATION
# ==========================================================

app = Flask(__name__)


# ==========================================================
# MODEL PATHS
# ==========================================================

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


# ==========================================================
# LOAD MODEL AND VECTORIZER
# ==========================================================

try:

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("Sentiment model loaded successfully.")
    print("TF-IDF vectorizer loaded successfully.")

except Exception as e:

    model = None
    vectorizer = None

    print("ERROR loading model/vectorizer:")
    print(e)


# ==========================================================
# DATABASE CONNECTION
# ==========================================================

def get_db_connection():

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise Exception("DATABASE_URL is not set.")

    connection = psycopg2.connect(database_url)

    return connection


# ==========================================================
# CREATE / UPDATE DATABASE TABLE
# ==========================================================

def setup_database():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor()

        # --------------------------------------------------
        # Create table if it does not exist
        # --------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (

                id SERIAL PRIMARY KEY,

                tweet_text TEXT NOT NULL,

                predicted_sentiment VARCHAR(20) NOT NULL,

                actual_sentiment VARCHAR(20),

                confidence DECIMAL(10, 4),

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

            );
        """)

        # --------------------------------------------------
        # Add columns if old table already exists
        # --------------------------------------------------

        cursor.execute("""
            ALTER TABLE predictions
            ADD COLUMN IF NOT EXISTS predicted_sentiment VARCHAR(20);
        """)

        cursor.execute("""
            ALTER TABLE predictions
            ADD COLUMN IF NOT EXISTS actual_sentiment VARCHAR(20);
        """)

        cursor.execute("""
            ALTER TABLE predictions
            ADD COLUMN IF NOT EXISTS confidence DECIMAL(10, 4);
        """)

        cursor.execute("""
            ALTER TABLE predictions
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP;
        """)

        # --------------------------------------------------
        # Check whether old "sentiment" column exists
        # --------------------------------------------------

        cursor.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'predictions'
                AND column_name = 'sentiment'
            );
        """)

        old_sentiment_exists = cursor.fetchone()[0]

        # --------------------------------------------------
        # If old table has "sentiment", migrate it safely
        # --------------------------------------------------

        if old_sentiment_exists:

            cursor.execute("""
                UPDATE predictions
                SET predicted_sentiment = sentiment
                WHERE predicted_sentiment IS NULL
                AND sentiment IS NOT NULL;
            """)

            print("Old sentiment data migrated.")

        # --------------------------------------------------
        # Give existing rows a current timestamp if missing
        # --------------------------------------------------

        cursor.execute("""
            UPDATE predictions
            SET created_at = CURRENT_TIMESTAMP
            WHERE created_at IS NULL;
        """)

        connection.commit()

        print("PostgreSQL database/table ready.")

    except Exception as e:

        print("Database setup error:")
        print(e)

        if connection:
            connection.rollback()

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ==========================================================
# RUN DATABASE SETUP
# ==========================================================

try:

    setup_database()

except Exception as e:

    print("Database initialization failed:")
    print(e)


# ==========================================================
# CALCULATE ANALYTICS
# ==========================================================

def calculate_analytics():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(
            cursor_factory=RealDictCursor
        )

        # --------------------------------------------------
        # Get only labelled predictions
        # --------------------------------------------------

        cursor.execute("""
            SELECT
                predicted_sentiment,
                actual_sentiment
            FROM predictions

            WHERE actual_sentiment IS NOT NULL
            AND actual_sentiment <> ''

            ORDER BY id ASC;
        """)

        rows = cursor.fetchall()

        # --------------------------------------------------
        # No labelled predictions
        # --------------------------------------------------

        if not rows:

            return {
                "labelled_count": 0,
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0
            }

        # --------------------------------------------------
        # Prepare data
        # --------------------------------------------------

        y_true = []
        y_pred = []

        for row in rows:

            actual = str(
                row["actual_sentiment"]
            ).strip().capitalize()

            predicted = str(
                row["predicted_sentiment"]
            ).strip().capitalize()

            if actual in [
                "Positive",
                "Negative",
                "Neutral"
            ]:

                y_true.append(actual)
                y_pred.append(predicted)

        # --------------------------------------------------
        # Safety check
        # --------------------------------------------------

        if not y_true:

            return {
                "labelled_count": 0,
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0
            }

        # --------------------------------------------------
        # Calculate Accuracy
        # --------------------------------------------------

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        # --------------------------------------------------
        # Calculate Macro Precision
        # --------------------------------------------------

        precision = precision_score(
            y_true,
            y_pred,
            labels=[
                "Positive",
                "Negative",
                "Neutral"
            ],
            average="macro",
            zero_division=0
        )

        # --------------------------------------------------
        # Calculate Macro Recall
        # --------------------------------------------------

        recall = recall_score(
            y_true,
            y_pred,
            labels=[
                "Positive",
                "Negative",
                "Neutral"
            ],
            average="macro",
            zero_division=0
        )

        # --------------------------------------------------
        # Calculate Macro F1
        # --------------------------------------------------

        f1 = f1_score(
            y_true,
            y_pred,
            labels=[
                "Positive",
                "Negative",
                "Neutral"
            ],
            average="macro",
            zero_division=0
        )

        return {

            "labelled_count": len(y_true),

            "accuracy":
                round(accuracy * 100, 2),

            "precision":
                round(precision * 100, 2),

            "recall":
                round(recall * 100, 2),

            "f1_score":
                round(f1 * 100, 2)

        }

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================================
# DASHBOARD PAGE
# ==========================================================

@app.route("/dashboard")
def dashboard():

    return render_template("dashboard.html")


# ==========================================================
# ANALYZE / PREDICT
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # --------------------------------------------------
        # Check model
        # --------------------------------------------------

        if model is None or vectorizer is None:

            return jsonify({

                "success": False,

                "error":
                    "Sentiment model is not loaded."

            }), 500

        # --------------------------------------------------
        # Get JSON
        # --------------------------------------------------

        data = request.get_json()

        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No data received."

            }), 400

        tweet_text = data.get(
            "text",
            ""
        ).strip()

        actual_sentiment = data.get(
            "actual_sentiment",
            ""
        ).strip()

        # --------------------------------------------------
        # Validate tweet
        # --------------------------------------------------

        if not tweet_text:

            return jsonify({

                "success": False,

                "error":
                    "Please enter a tweet."

            }), 400

        # --------------------------------------------------
        # Actual sentiment is REQUIRED
        # --------------------------------------------------

        allowed_sentiments = [
            "Positive",
            "Negative",
            "Neutral"
        ]

        if not actual_sentiment:

            return jsonify({

                "success": False,

                "error":
                    "Please select the actual sentiment."

            }), 400

        actual_sentiment = (
            actual_sentiment.capitalize()
        )

        if actual_sentiment not in allowed_sentiments:

            return jsonify({

                "success": False,

                "error":
                    "Actual sentiment must be "
                    "Positive, Negative, or Neutral."

            }), 400

        # ==================================================
        # TRANSFORM TEXT
        # ==================================================

        text_vector = vectorizer.transform(
            [tweet_text]
        )

        # ==================================================
        # PREDICT SENTIMENT
        # ==================================================

        prediction = model.predict(
            text_vector
        )

        predicted_sentiment = prediction[0]

        if hasattr(
            predicted_sentiment,
            "item"
        ):

            predicted_sentiment = (
                predicted_sentiment.item()
            )

        predicted_sentiment = str(
            predicted_sentiment
        ).strip().capitalize()

        # ==================================================
        # CALCULATE CONFIDENCE
        # ==================================================

        confidence = 0.0

        try:

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = (
                    model.predict_proba(
                        text_vector
                    )
                )

                confidence = (
                    float(
                        max(probabilities[0])
                    ) * 100
                )

        except Exception as e:

            print(
                "Confidence calculation error:"
            )

            print(e)

            confidence = 0.0

        # ==================================================
        # SAVE TO POSTGRESQL
        # ==================================================

        connection = None
        cursor = None

        try:

            connection = get_db_connection()

            cursor = connection.cursor()

            cursor.execute("""
                INSERT INTO predictions
                (
                    tweet_text,
                    predicted_sentiment,
                    actual_sentiment,
                    confidence,
                    created_at
                )

                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    CURRENT_TIMESTAMP
                );
            """, (

                tweet_text,

                predicted_sentiment,

                actual_sentiment,

                confidence

            ))

            connection.commit()

        except Exception as e:

            print(
                "Database insert error:"
            )

            print(e)

            if connection:
                connection.rollback()

            return jsonify({

                "success": False,

                "error":
                    "Prediction was made, "
                    "but could not be saved."

            }), 500

        finally:

            if cursor:
                cursor.close()

            if connection:
                connection.close()

        # ==================================================
        # CALCULATE UPDATED METRICS
        # ==================================================

        analytics_data = calculate_analytics()

        # ==================================================
        # RETURN PREDICTION + METRICS
        # ==================================================

        return jsonify({

            "success": True,

            "tweet":
                tweet_text,

            "predicted_sentiment":
                predicted_sentiment,

            "actual_sentiment":
                actual_sentiment,

            "confidence":
                round(confidence, 2),

            # ----------------------------------------------
            # UPDATED METRICS
            # ----------------------------------------------

            "labelled_count":
                analytics_data[
                    "labelled_count"
                ],

            "accuracy":
                analytics_data[
                    "accuracy"
                ],

            "precision":
                analytics_data[
                    "precision"
                ],

            "recall":
                analytics_data[
                    "recall"
                ],

            "f1_score":
                analytics_data[
                    "f1_score"
                ]

        })

    except Exception as e:

        print(
            "Prediction error:"
        )

        print(e)

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ==========================================================
# GET PREDICTION HISTORY
# ==========================================================

@app.route("/api/history")
def prediction_history():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(
            cursor_factory=RealDictCursor
        )

        cursor.execute("""
            SELECT
                id,
                tweet_text,
                predicted_sentiment,
                actual_sentiment,
                confidence,
                created_at

            FROM predictions

            ORDER BY id DESC

            LIMIT 100;
        """)

        rows = cursor.fetchall()

        history = []

        for row in rows:

            item = dict(row)

            if item.get(
                "confidence"
            ) is not None:

                item["confidence"] = float(
                    item["confidence"]
                )

            if item.get(
                "created_at"
            ) is not None:

                item["created_at"] = (
                    item["created_at"]
                    .strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

            history.append(item)

        return jsonify(history)

    except Exception as e:

        print(
            "History error:"
        )

        print(e)

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ==========================================================
# ANALYTICS API
# ==========================================================

@app.route("/api/analytics")
def analytics():

    try:

        analytics_data = (
            calculate_analytics()
        )

        # --------------------------------------------------
        # Sentiment distribution
        # --------------------------------------------------

        connection = get_db_connection()

        cursor = connection.cursor(
            cursor_factory=RealDictCursor
        )

        cursor.execute("""
            SELECT
                actual_sentiment,
                COUNT(*) AS count

            FROM predictions

            WHERE actual_sentiment IS NOT NULL
            AND actual_sentiment <> ''

            GROUP BY actual_sentiment;
        """)

        rows = cursor.fetchall()

        cursor.close()
        connection.close()

        sentiment_counts = {

            "Positive": 0,

            "Negative": 0,

            "Neutral": 0

        }

        for row in rows:

            sentiment = str(
                row["actual_sentiment"]
            ).capitalize()

            if sentiment in sentiment_counts:

                sentiment_counts[
                    sentiment
                ] = int(row["count"])

        total = sum(
            sentiment_counts.values()
        )

        sentiment_distribution = {}

        for sentiment, count in (
            sentiment_counts.items()
        ):

            percentage = (
                (count / total) * 100
                if total > 0
                else 0
            )

            sentiment_distribution[
                sentiment
            ] = {

                "count": count,

                "percentage":
                    round(
                        percentage,
                        2
                    )

            }

        return jsonify({

            "success": True,

            "labelled_count":
                analytics_data[
                    "labelled_count"
                ],

            "accuracy":
                analytics_data[
                    "accuracy"
                ],

            "precision":
                analytics_data[
                    "precision"
                ],

            "recall":
                analytics_data[
                    "recall"
                ],

            "f1_score":
                analytics_data[
                    "f1_score"
                ],

            "sentiment_distribution":
                sentiment_distribution

        })

    except Exception as e:

        print(
            "Analytics error:"
        )

        print(e)

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ==========================================================
# DELETE ALL PREDICTION HISTORY
# ==========================================================

@app.route(
    "/api/history/delete",
    methods=["DELETE"]
)
def delete_history():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute("""
            DELETE FROM predictions;
        """)

        connection.commit()

        return jsonify({

            "success": True,

            "message":
                "Prediction history deleted."

        })

    except Exception as e:

        print(
            "Delete history error:"
        )

        print(e)

        if connection:
            connection.rollback()

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "ok",

        "model_loaded":
            model is not None,

        "vectorizer_loaded":
            vectorizer is not None

    })


# ==========================================================
# APPLICATION START
# ==========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),

        debug=False

    )