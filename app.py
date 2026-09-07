import os
import re
from datetime import timezone, timedelta

from flask import Flask, jsonify, render_template, request
import joblib

from db.db import get_db_connection, setup_database


# ==========================================================
# FLASK APPLICATION
# ==========================================================

app = Flask(__name__)


# ==========================================================
# DATABASE SETUP
# ==========================================================

try:
    setup_database()
except Exception as e:
    print("Database setup failed:", e)


# ==========================================================
# PAGE ROUTES
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ==========================================================
# MODEL PATHS & LOADING
# ==========================================================

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

print("Loading sentiment model and vectorizer...")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("All models loaded successfully!")


# ==========================================================
# MYANMAR TIMEZONE
# ==========================================================

MYANMAR_TZ = timezone(timedelta(hours=6, minutes=30))


# ==========================================================
# PREDICTION API
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    conn = None
    cur = None

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid request."
            }), 400

        text = data.get("text", "").strip()

        if not text:
            return jsonify({
                "success": False,
                "error": "Please enter a tweet."
            }), 400


        # ==================================================
        # TEXT PREPROCESSING
        # ==================================================

        cleaned_text = text.lower()

        cleaned_text = re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            cleaned_text
        )

        cleaned_text = re.sub(
            r"@\w+",
            " ",
            cleaned_text
        )

        cleaned_text = re.sub(
            r"#(\w+)",
            r"\1",
            cleaned_text
        )

        cleaned_text = (
            cleaned_text
            .replace("can't", "cannot")
            .replace("won't", "will not")
        )

        cleaned_text = re.sub(
            r"n't\b",
            " not",
            cleaned_text
        )

        cleaned_text = re.sub(
            r"[^a-zA-Z\s]",
            " ",
            cleaned_text
        )

        cleaned_text = re.sub(
            r"\s+",
            " ",
            cleaned_text
        ).strip()


        # ==================================================
        # MODEL INFERENCE
        # ==================================================

        features = vectorizer.transform([cleaned_text])

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]

        sentiment = str(prediction).capitalize()

        # Convert NumPy float64 to normal Python float
        confidence_percentage = float(
            round(float(max(probabilities)) * 100, 2)
        )


        # ==================================================
        # POSTGRESQL INSERTION
        # ==================================================

        conn = get_db_connection()
        cur = conn.cursor()

        insert_query = """
            INSERT INTO predictions
            (tweet_text, sentiment, confidence)
            VALUES (%s, %s, %s)
        """

        cur.execute(
            insert_query,
            (
                text,
                sentiment,
                confidence_percentage
            )
        )

        conn.commit()


        # ==================================================
        # RESPONSE
        # ==================================================

        return jsonify({
            "success": True,
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence_percentage
        })


    except Exception as e:

        if conn:
            conn.rollback()

        print("Prediction error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


    finally:

        if cur:
            cur.close()

        if conn:
            conn.close()


# ==========================================================
# HISTORY API
# ==========================================================

@app.route("/api/history")
def history():

    conn = None
    cur = None

    try:

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                id,
                tweet_text,
                sentiment,
                confidence,
                created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT 50
            """
        )

        rows = cur.fetchall()

        history_data = []


        # ==================================================
        # CONVERT DATABASE TIME TO MYANMAR TIME
        # ==================================================

        for row in rows:

            created_at = row[4]

            if created_at:

                # If PostgreSQL returns a timezone-naive datetime,
                # assume it is UTC (Render/PostgreSQL commonly uses UTC).
                if created_at.tzinfo is None:

                    created_at = created_at.replace(
                        tzinfo=timezone.utc
                    )

                # Convert UTC to Myanmar Time (UTC+6:30)
                created_at = created_at.astimezone(
                    MYANMAR_TZ
                )

                created_at_string = created_at.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            else:

                created_at_string = ""


            history_data.append({

                "id": row[0],

                "tweet": row[1],

                "sentiment": row[2],

                "confidence":
                    float(row[3])
                    if row[3] is not None
                    else 0,

                "created_at":
                    created_at_string
            })


        return jsonify(history_data)


    except Exception as e:

        print("History error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


    finally:

        if cur:
            cur.close()

        if conn:
            conn.close()


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )