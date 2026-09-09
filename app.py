import os
import re
from datetime import timezone, timedelta

import joblib
import pandas as pd

from flask import Flask, jsonify, render_template, request

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
    print("Database setup completed successfully.")
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
# MODEL PATHS
# ==========================================================

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

print("Loading sentiment model and vectorizer...")

try:

    model = joblib.load(MODEL_PATH)

    vectorizer = joblib.load(VECTORIZER_PATH)

    print("All models loaded successfully!")

except Exception as e:

    print("Model loading failed:", e)

    model = None
    vectorizer = None


# ==========================================================
# KAGGLE DATASET
# ==========================================================

DATASET_PATH = "tweet.csv"

print("Loading Kaggle Twitter dataset...")

try:

    kaggle_df = pd.read_csv(DATASET_PATH)

    print(
        f"Kaggle dataset loaded successfully: "
        f"{len(kaggle_df)} rows"
    )

    print(
        "Dataset columns:",
        list(kaggle_df.columns)
    )

    # ------------------------------------------------------
    # Keep only the columns needed for analytics
    # ------------------------------------------------------

    required_columns = [
        "textID",
        "text",
        "selected_text",
        "sentiment"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in kaggle_df.columns
    ]

    if missing_columns:

        print(
            "Missing dataset columns:",
            missing_columns
        )

        kaggle_df = pd.DataFrame()

    else:

        # --------------------------------------------------
        # Remove rows without sentiment
        # --------------------------------------------------

        kaggle_df = kaggle_df.dropna(
            subset=["sentiment"]
        )

        # --------------------------------------------------
        # Remove rows without text
        # --------------------------------------------------

        kaggle_df = kaggle_df.dropna(
            subset=["text"]
        )

        # --------------------------------------------------
        # Remove duplicate tweets
        # --------------------------------------------------

        kaggle_df = kaggle_df.drop_duplicates(
            subset=["textID"]
        )

        # --------------------------------------------------
        # Clean sentiment values
        # --------------------------------------------------

        kaggle_df["sentiment"] = (
            kaggle_df["sentiment"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # --------------------------------------------------
        # Calculate tweet length
        # --------------------------------------------------

        kaggle_df["tweet_length"] = (
            kaggle_df["text"]
            .astype(str)
            .str.len()
        )

        print(
            "Kaggle dataset preprocessing completed."
        )

except Exception as e:

    print(
        "Kaggle dataset loading failed:",
        e
    )

    kaggle_df = pd.DataFrame()


# ==========================================================
# MYANMAR TIMEZONE
# ==========================================================

MYANMAR_TZ = timezone(
    timedelta(
        hours=6,
        minutes=30
    )
)


# ==========================================================
# PREDICTION API
# ==========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    conn = None
    cur = None

    try:

        # ==================================================
        # REQUEST DATA
        # ==================================================

        data = request.get_json()

        if not data:

            return jsonify({

                "success": False,

                "error":
                    "Invalid request."

            }), 400


        text = data.get(
            "text",
            ""
        ).strip()


        if not text:

            return jsonify({

                "success": False,

                "error":
                    "Please enter a tweet."

            }), 400


        # ==================================================
        # CHECK MODEL
        # ==================================================

        if model is None or vectorizer is None:

            return jsonify({

                "success": False,

                "error":
                    "Sentiment model is not available."

            }), 500


        # ==================================================
        # TEXT PREPROCESSING
        # ==================================================

        cleaned_text = text.lower()


        # Remove URLs

        cleaned_text = re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            cleaned_text
        )


        # Remove mentions

        cleaned_text = re.sub(
            r"@\w+",
            " ",
            cleaned_text
        )


        # Keep hashtag word

        cleaned_text = re.sub(
            r"#(\w+)",
            r"\1",
            cleaned_text
        )


        # Expand common contractions

        cleaned_text = (
            cleaned_text
            .replace(
                "can't",
                "cannot"
            )
            .replace(
                "won't",
                "will not"
            )
        )


        cleaned_text = re.sub(
            r"n't\b",
            " not",
            cleaned_text
        )


        # Keep English letters and spaces

        cleaned_text = re.sub(
            r"[^a-zA-Z\s]",
            " ",
            cleaned_text
        )


        # Remove extra spaces

        cleaned_text = re.sub(
            r"\s+",
            " ",
            cleaned_text
        ).strip()


        # ==================================================
        # MODEL INFERENCE
        # ==================================================

        features = vectorizer.transform(
            [cleaned_text]
        )


        prediction = model.predict(
            features
        )[0]


        probabilities = model.predict_proba(
            features
        )[0]


        sentiment = str(
            prediction
        ).capitalize()


        # ==================================================
        # CONFIDENCE
        # ==================================================

        confidence_percentage = float(
            round(
                float(
                    max(probabilities)
                ) * 100,
                2
            )
        )


        # ==================================================
        # SAVE PREDICTION TO POSTGRESQL
        # ==================================================

        conn = get_db_connection()

        cur = conn.cursor()


        insert_query = """
            INSERT INTO predictions
            (
                tweet_text,
                sentiment,
                confidence
            )
            VALUES
            (
                %s,
                %s,
                %s
            )
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

            "confidence":
                confidence_percentage

        })


    except Exception as e:

        if conn:

            conn.rollback()


        print(
            "Prediction error:",
            e
        )


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

                if created_at.tzinfo is None:

                    created_at = created_at.replace(
                        tzinfo=timezone.utc
                    )


                created_at = created_at.astimezone(
                    MYANMAR_TZ
                )


                created_at_string = (
                    created_at.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
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


        return jsonify(
            history_data
        )


    except Exception as e:

        print(
            "History error:",
            e
        )


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
# KAGGLE DATASET ANALYTICS API
# ==========================================================

@app.route("/api/analytics")
def analytics():

    try:

        # ==================================================
        # CHECK DATASET
        # ==================================================

        if kaggle_df.empty:

            return jsonify({

                "success": False,

                "error":
                    "Kaggle dataset is not available."

            }), 500


        # ==================================================
        # TOTAL TWEETS
        # ==================================================

        total_tweets = int(
            len(kaggle_df)
        )


        # ==================================================
        # SENTIMENT COUNTS
        # ==================================================

        sentiment_counts = (
            kaggle_df["sentiment"]
            .value_counts()
        )


        positive = int(
            sentiment_counts.get(
                "positive",
                0
            )
        )


        negative = int(
            sentiment_counts.get(
                "negative",
                0
            )
        )


        neutral = int(
            sentiment_counts.get(
                "neutral",
                0
            )
        )


        # ==================================================
        # SENTIMENT PERCENTAGES
        # ==================================================

        if total_tweets > 0:

            positive_percentage = round(
                positive /
                total_tweets *
                100,
                2
            )

            negative_percentage = round(
                negative /
                total_tweets *
                100,
                2
            )

            neutral_percentage = round(
                neutral /
                total_tweets *
                100,
                2
            )

        else:

            positive_percentage = 0

            negative_percentage = 0

            neutral_percentage = 0


        # ==================================================
        # AVERAGE TWEET LENGTH
        # ==================================================

        average_tweet_length = round(
            float(
                kaggle_df[
                    "tweet_length"
                ].mean()
            ),
            2
        )


        # ==================================================
        # AVERAGE LENGTH BY SENTIMENT
        # ==================================================

        length_by_sentiment = (
            kaggle_df
            .groupby("sentiment")[
                "tweet_length"
            ]
            .mean()
            .round(2)
            .to_dict()
        )


        # ==================================================
        # RETURN ANALYTICS
        # ==================================================

        return jsonify({

            "success": True,

            "data_source":
                "Kaggle Twitter Sentiment Dataset",

            "total_tweets":
                total_tweets,

            "positive":
                positive,

            "negative":
                negative,

            "neutral":
                neutral,

            "positive_percentage":
                positive_percentage,

            "negative_percentage":
                negative_percentage,

            "neutral_percentage":
                neutral_percentage,

            "average_tweet_length":
                average_tweet_length,

            "average_length_positive":
                float(
                    length_by_sentiment.get(
                        "positive",
                        0
                    )
                ),

            "average_length_negative":
                float(
                    length_by_sentiment.get(
                        "negative",
                        0
                    )
                ),

            "average_length_neutral":
                float(
                    length_by_sentiment.get(
                        "neutral",
                        0
                    )
                )

        })


    except Exception as e:

        print(
            "Analytics error:",
            e
        )


        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ==========================================================
# MODEL PERFORMANCE API
# ==========================================================

@app.route("/api/model-performance")
def model_performance():

    return jsonify({

        "success": True,

        "accuracy": 68.12,

        "precision": 68.20,

        "recall": 68.12,

        "f1_score": 68.15

    })


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )
