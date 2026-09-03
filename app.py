import os
import re
from flask import Flask, jsonify, render_template, request
import joblib
import mysql.connector

# ==========================================================
# FLASK APPLICATION
# ==========================================================

app = Flask(__name__)


# ==========================================================
# DATABASE (MySQL Setup)
# ==========================================================

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "root")
DB_NAME = os.environ.get("DB_NAME", "Cloud-Twitter Sentiment")
DB_PORT = int(os.environ.get("DB_PORT", 3306))


def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
    )


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
# PREDICTION API
# ==========================================================


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"success": False, "error": "Please enter a tweet."}), 400

        # Text Preprocessing
        cleaned_text = text.lower()
        cleaned_text = re.sub(r"https?://\S+|www\.\S+", " ", cleaned_text)
        cleaned_text = re.sub(r"@\w+", " ", cleaned_text)
        cleaned_text = re.sub(r"#(\w+)", r"\1", cleaned_text)
        cleaned_text = cleaned_text.replace("can't", "cannot").replace("won't", "will not")
        cleaned_text = re.sub(r"n't\b", " not", cleaned_text)
        cleaned_text = re.sub(r"[^a-zA-Z\s]", " ", cleaned_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        # Model Inference
        features = vectorizer.transform([cleaned_text])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        sentiment = str(prediction).capitalize()
        confidence_percentage = round(max(probabilities) * 100, 2)

        # MySQL Insertion
        conn = get_db_connection()
        cur = conn.cursor()

        insert_query = """
            INSERT INTO predictions (tweet_text, sentiment, confidence)
            VALUES (%s, %s, %s)
        """
        cur.execute(insert_query, (text, sentiment, confidence_percentage))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "success": True,
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence_percentage,
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ==========================================================
# HISTORY API
# ==========================================================


@app.route("/api/history")
def history():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, tweet_text, sentiment, confidence, created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT 50
            """
        )

        rows = cur.fetchall()
        cur.close()
        conn.close()

        history_data = []
        for row in rows:
            history_data.append({
                "id": row[0],
                "tweet": row[1],
                "sentiment": row[2],
                "confidence": float(row[3]) if row[3] is not None else 0,
                "created_at": row[4].strftime("%Y-%m-%d %H:%M:%S") if row[4] else "",
            })

        return jsonify(history_data)

    except Exception as e:
        print("History error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)