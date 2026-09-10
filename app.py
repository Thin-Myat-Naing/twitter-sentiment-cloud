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
    """
    Connect to Render PostgreSQL using DATABASE_URL.
    """

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
        # Add new columns if the old table already exists
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
        # If old table used "sentiment", copy it to
        # predicted_sentiment where necessary.
        # --------------------------------------------------

        cursor.execute("""
            UPDATE predictions
            SET predicted_sentiment = sentiment
            WHERE predicted_sentiment IS NULL
              AND sentiment IS NOT NULL;
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
                "error": "Sentiment model is not loaded."
            }), 500


        # --------------------------------------------------
        # Get JSON data
        # --------------------------------------------------

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "error": "No data received."
            }), 400


        tweet_text = data.get("text", "").strip()

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
                "error": "Please enter a tweet."
            }), 400


        # --------------------------------------------------
        # Validate actual sentiment
        #
        # It is OPTIONAL.
        # --------------------------------------------------

        allowed_sentiments = [
            "Positive",
            "Negative",
            "Neutral"
        ]

        if actual_sentiment:

            # Normalize first letter/case

            actual_sentiment = actual_sentiment.capitalize()

            if actual_sentiment not in allowed_sentiments:

                return jsonify({
                    "success": False,
                    "error": (
                        "Actual sentiment must be "
                        "Positive, Negative, or Neutral."
                    )
                }), 400

        else:

            actual_sentiment = None


        # ==================================================
        # TRANSFORM TEXT
        # ==================================================

        text_vector = vectorizer.transform([tweet_text])


        # ==================================================
        # PREDICT SENTIMENT
        # ==================================================

        prediction = model.predict(text_vector)

        predicted_sentiment = prediction[0]


        # --------------------------------------------------
        # Convert prediction to normal string
        # --------------------------------------------------

        if hasattr(predicted_sentiment, "item"):

            predicted_sentiment = predicted_sentiment.item()

        predicted_sentiment = str(
            predicted_sentiment
        ).capitalize()


        # ==================================================
        # CALCULATE CONFIDENCE
        # ==================================================

        confidence = 0.0


        try:

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    text_vector
                )

                confidence = float(
                    max(probabilities[0])
                ) * 100

            else:

                confidence = 0.0

        except Exception as e:

            print("Confidence calculation error:")
            print(e)

            confidence = 0.0


        # ==================================================
        # SAVE PREDICTION TO POSTGRESQL
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
                    confidence
                )
                VALUES (%s, %s, %s, %s)
            """, (
                tweet_text,
                predicted_sentiment,
                actual_sentiment,
                confidence
            ))

            connection.commit()

        except Exception as e:

            print("Database insert error:")
            print(e)

            if connection:
                connection.rollback()

            return jsonify({
                "success": False,
                "error": "Prediction was made, but could not be saved."
            }), 500

        finally:

            if cursor:
                cursor.close()

            if connection:
                connection.close()


        # ==================================================
        # RETURN RESULT
        # ==================================================

        return jsonify({

            "success": True,

            "tweet": tweet_text,

            "predicted_sentiment":
                predicted_sentiment,

            "actual_sentiment":
                actual_sentiment,

            "confidence":
                round(confidence, 2)

        })


    except Exception as e:

        print("Prediction error:")
        print(e)

        return jsonify({
            "success": False,
            "error": str(e)
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


        # --------------------------------------------------
        # Convert PostgreSQL data into JSON-safe format
        # --------------------------------------------------

        history = []

        for row in rows:

            item = dict(row)

            if item.get("confidence") is not None:

                item["confidence"] = float(
                    item["confidence"]
                )

            if item.get("created_at") is not None:

                item["created_at"] = (
                    item["created_at"]
                    .strftime("%Y-%m-%d %H:%M:%S")
                )

            history.append(item)


        return jsonify(history)


    except Exception as e:

        print("History error:")
        print(e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ==========================================================
# ANALYTICS / MODEL EVALUATION
# ==========================================================

@app.route("/api/analytics")
def analytics():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(
            cursor_factory=RealDictCursor
        )


        # --------------------------------------------------
        # Only use records where the user provided
        # an actual sentiment.
        # --------------------------------------------------

        cursor.execute("""
            SELECT
                predicted_sentiment,
                actual_sentiment
            FROM predictions
            WHERE actual_sentiment IS NOT NULL
              AND actual_sentiment <> '';
        """)

        rows = cursor.fetchall()


        # --------------------------------------------------
        # No labelled data yet
        # --------------------------------------------------

        if not rows:

            return jsonify({

                "success": True,

                "labelled_count": 0,

                "accuracy": 0,

                "precision": 0,

                "recall": 0,

                "f1_score": 0,

                "message":
                    "Please provide actual sentiment for predictions."

            })


        # --------------------------------------------------
        # Prepare actual and predicted labels
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


            y_true.append(actual)
            y_pred.append(predicted)


        # ==================================================
        # CALCULATE METRICS
        # ==================================================

        accuracy = accuracy_score(
            y_true,
            y_pred
        )


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


        # ==================================================
        # SENTIMENT DISTRIBUTION
        # ==================================================

        sentiment_counts = {
            "Positive": 0,
            "Negative": 0,
            "Neutral": 0
        }


        for sentiment in y_true:

            if sentiment in sentiment_counts:

                sentiment_counts[sentiment] += 1


        total = len(y_true)


        sentiment_distribution = {}

        for sentiment, count in sentiment_counts.items():

            if total > 0:

                percentage = (
                    count / total
                ) * 100

            else:

                percentage = 0


            sentiment_distribution[sentiment] = {

                "count": count,

                "percentage":
                    round(percentage, 2)

            }


        # ==================================================
        # RETURN ANALYTICS
        # ==================================================

        return jsonify({

            "success": True,

            "labelled_count": total,

            "accuracy":
                round(accuracy * 100, 2),

            "precision":
                round(precision * 100, 2),

            "recall":
                round(recall * 100, 2),

            "f1_score":
                round(f1 * 100, 2),

            "sentiment_distribution":
                sentiment_distribution

        })


    except Exception as e:

        print("Analytics error:")
        print(e)

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ==========================================================
# DELETE ALL PREDICTION HISTORY
# ==========================================================

@app.route("/api/history/delete", methods=["DELETE"])
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

        print("Delete history error:")
        print(e)

        if connection:
            connection.rollback()


        return jsonify({

            "success": False,

            "error": str(e)

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

        "status": "ok",

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