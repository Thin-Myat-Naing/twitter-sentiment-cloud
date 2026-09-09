import os
import re
from datetime import timezone, timedelta

import joblib
import psycopg2

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

    print(
        "Database setup completed successfully."
    )

except Exception as e:

    print(
        "Database setup failed:",
        e
    )


# ==========================================================
# PAGE ROUTES
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# ==========================================================
# MODEL PATHS
# ==========================================================

MODEL_PATH = "models/sentiment_model.pkl"

VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


print(
    "Loading sentiment model and vectorizer..."
)


# ==========================================================
# LOAD MODEL AND VECTORIZER
# ==========================================================

try:

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    print(
        "All models loaded successfully!"
    )

except Exception as e:

    print(
        "Model loading failed:",
        e
    )

    model = None

    vectorizer = None


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


        # ==================================================
        # VALIDATE TEXT
        # ==================================================

        if not text:

            return jsonify({

                "success": False,

                "error":
                    "Please enter a tweet."

            }), 400


        # ==================================================
        # CHECK MODEL
        # ==================================================

        if (
            model is None
            or
            vectorizer is None
        ):

            return jsonify({

                "success": False,

                "error":
                    "Sentiment model is not available."

            }), 500


        # ==================================================
        # TEXT PREPROCESSING
        # ==========================================================

        cleaned_text = text.lower()


        # --------------------------------------------------
        # Remove URLs
        # --------------------------------------------------

        cleaned_text = re.sub(
            r"https?://\S+|www\.\S+",
            " ",
            cleaned_text
        )


        # --------------------------------------------------
        # Remove mentions
        # --------------------------------------------------

        cleaned_text = re.sub(
            r"@\w+",
            " ",
            cleaned_text
        )


        # --------------------------------------------------
        # Keep hashtag word
        # --------------------------------------------------

        cleaned_text = re.sub(
            r"#(\w+)",
            r"\1",
            cleaned_text
        )


        # --------------------------------------------------
        # Expand common contractions
        # --------------------------------------------------

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


        # --------------------------------------------------
        # Keep English letters and spaces
        # --------------------------------------------------

        cleaned_text = re.sub(
            r"[^a-zA-Z\s]",
            " ",
            cleaned_text
        )


        # --------------------------------------------------
        # Remove extra spaces
        # --------------------------------------------------

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


        # ==================================================
        # SENTIMENT
        # ==================================================

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
        # RETURN PREDICTION
        # ==================================================

        return jsonify({

            "success": True,

            "text": text,

            "sentiment": sentiment,

            "confidence":
                confidence_percentage

        })


    except Exception as e:

        # ==================================================
        # ROLLBACK DATABASE
        # ==================================================

        if conn:

            conn.rollback()


        print(
            "Prediction error:",
            e
        )


        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


    finally:

        # ==================================================
        # CLOSE DATABASE
        # ==================================================

        if cur:

            cur.close()


        if conn:

            conn.close()


# ==========================================================
# PREDICTION HISTORY API
# ==========================================================

@app.route(
    "/api/history"
)
def history():

    conn = None

    cur = None

    try:

        # ==================================================
        # DATABASE CONNECTION
        # ==================================================

        conn = get_db_connection()

        cur = conn.cursor()


        # ==================================================
        # GET RECENT PREDICTIONS
        # ==================================================

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
        # CONVERT TIME TO MYANMAR TIME
        # ==================================================

        for row in rows:

            created_at = row[4]


            if created_at:

                # ------------------------------------------
                # If PostgreSQL returns a naive datetime,
                # treat it as UTC.
                # ------------------------------------------

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

                "id":
                    row[0],

                "tweet":
                    row[1],

                "sentiment":
                    row[2],

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

            "error":
                str(e)

        }), 500


    finally:

        # ==================================================
        # CLOSE DATABASE
        # ==================================================

        if cur:

            cur.close()


        if conn:

            conn.close()


# ==========================================================
# LIVE PREDICTION ANALYTICS API
# ==========================================================

@app.route(
    "/api/analytics"
)
def analytics():

    conn = None

    cur = None

    try:

        # ==================================================
        # DATABASE CONNECTION
        # ==================================================

        conn = get_db_connection()

        cur = conn.cursor()


        # ==================================================
        # TOTAL TWEETS
        # ==================================================

        cur.execute(
            """

            SELECT COUNT(*)

            FROM predictions

            """
        )


        total_result = cur.fetchone()


        total_tweets = (
            total_result[0]
            if total_result
            else 0
        )


        # ==================================================
        # SENTIMENT COUNTS
        # ==================================================

        cur.execute(
            """

            SELECT

                COUNT(*) FILTER (
                    WHERE LOWER(sentiment) = 'positive'
                ) AS positive,

                COUNT(*) FILTER (
                    WHERE LOWER(sentiment) = 'negative'
                ) AS negative,

                COUNT(*) FILTER (
                    WHERE LOWER(sentiment) = 'neutral'
                ) AS neutral

            FROM predictions

            """
        )


        sentiment_row = cur.fetchone()


        positive = (
            sentiment_row[0]
            if sentiment_row
            and sentiment_row[0] is not None
            else 0
        )


        negative = (
            sentiment_row[1]
            if sentiment_row
            and sentiment_row[1] is not None
            else 0
        )


        neutral = (
            sentiment_row[2]
            if sentiment_row
            and sentiment_row[2] is not None
            else 0
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

        cur.execute(
            """

            SELECT

                COALESCE(
                    ROUND(
                        AVG(
                            LENGTH(tweet_text)
                        ),
                        2
                    ),
                    0
                )

            FROM predictions

            """
        )


        average_result = cur.fetchone()


        average_tweet_length = (
            average_result[0]
            if average_result
            and average_result[0] is not None
            else 0
        )


        # ==================================================
        # AVERAGE LENGTH BY SENTIMENT
        # ==================================================

        cur.execute(
            """

            SELECT

                COALESCE(
                    ROUND(
                        AVG(
                            LENGTH(tweet_text)
                        ) FILTER (
                            WHERE LOWER(sentiment) = 'positive'
                        ),
                        2
                    ),
                    0
                ) AS positive_length,


                COALESCE(
                    ROUND(
                        AVG(
                            LENGTH(tweet_text)
                        ) FILTER (
                            WHERE LOWER(sentiment) = 'negative'
                        ),
                        2
                    ),
                    0
                ) AS negative_length,


                COALESCE(
                    ROUND(
                        AVG(
                            LENGTH(tweet_text)
                        ) FILTER (
                            WHERE LOWER(sentiment) = 'neutral'
                        ),
                        2
                    ),
                    0
                ) AS neutral_length

            FROM predictions

            """
        )


        length_row = cur.fetchone()


        average_positive_length = (
            length_row[0]
            if length_row
            and length_row[0] is not None
            else 0
        )


        average_negative_length = (
            length_row[1]
            if length_row
            and length_row[1] is not None
            else 0
        )


        average_neutral_length = (
            length_row[2]
            if length_row
            and length_row[2] is not None
            else 0
        )


        # ==================================================
        # RETURN ANALYTICS
        # ==================================================

        return jsonify({

            "success": True,

            "data_source":
                "User Submitted Tweets",

            "total_tweets":
                int(total_tweets),

            "positive":
                int(positive),

            "negative":
                int(negative),

            "neutral":
                int(neutral),

            "positive_percentage":
                float(
                    positive_percentage
                ),

            "negative_percentage":
                float(
                    negative_percentage
                ),

            "neutral_percentage":
                float(
                    neutral_percentage
                ),

            "average_tweet_length":
                float(
                    average_tweet_length
                ),

            "average_length_positive":
                float(
                    average_positive_length
                ),

            "average_length_negative":
                float(
                    average_negative_length
                ),

            "average_length_neutral":
                float(
                    average_neutral_length
                )

        })


    except Exception as e:

        print(
            "Analytics error:",
            e
        )


        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


    finally:

        # ==================================================
        # CLOSE DATABASE
        # ==================================================

        if cur:

            cur.close()


        if conn:

            conn.close()


# ==========================================================
# MODEL PERFORMANCE API
# ==========================================================

@app.route(
    "/api/model-performance"
)
def model_performance():

    # ======================================================
    # MODEL EVALUATION METRICS
    #
    # These values come from the trained model's
    # evaluation on the test dataset.
    #
    # They should NOT be calculated from individual
    # tweets entered by users.
    # ======================================================

    return jsonify({

        "success": True,

        "accuracy":
            68.12,

        "precision":
            68.20,

        "recall":
            68.12,

        "f1_score":
            68.15

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
