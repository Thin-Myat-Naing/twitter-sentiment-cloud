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
# KAGGLE DATASET
# ==========================================================

DATASET_PATH = "tweet.csv"


print(
    "Loading Kaggle Twitter dataset..."
)


try:

    kaggle_df = pd.read_csv(
        DATASET_PATH
    )


    print(
        f"Kaggle dataset loaded successfully: "
        f"{len(kaggle_df)} rows"
    )


    print(
        "Dataset columns:",
        list(kaggle_df.columns)
    )


    # ------------------------------------------------------
    # Required columns
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


        print(
            "Kaggle sentiment distribution:"
        )

        print(
            kaggle_df["sentiment"].value_counts()
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
        # ==================================================

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

            "text":
                text,

            "sentiment":
                sentiment,

            "confidence":
                confidence_percentage

        })


    except Exception as e:

        # ==================================================
        # DATABASE ROLLBACK
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
        # CLOSE DATABASE CONNECTION
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

        if cur:

            cur.close()


        if conn:

            conn.close()


# ==========================================================
# COMBINED DATA ANALYTICS API
# ==========================================================
#
# Dashboard analytics are calculated from:
#
#     1. Kaggle dataset
#     2. User-submitted predictions
#
# Therefore:
#
#     Total Tweets =
#         Kaggle Tweets + User Tweets
#
#     Positive =
#         Kaggle Positive + User Positive
#
#     Negative =
#         Kaggle Negative + User Negative
#
#     Neutral =
#         Kaggle Neutral + User Neutral
#
# ==========================================================

@app.route(
    "/api/analytics"
)
def analytics():

    conn = None

    cur = None


    try:

        # ==================================================
        # KAGGLE DATASET ANALYTICS
        # ==================================================

        if kaggle_df.empty:

            kaggle_total = 0

            kaggle_positive = 0

            kaggle_negative = 0

            kaggle_neutral = 0

            kaggle_average_length = 0

            kaggle_positive_length = 0

            kaggle_negative_length = 0

            kaggle_neutral_length = 0


        else:

            # ----------------------------------------------
            # Total Kaggle tweets
            # ----------------------------------------------

            kaggle_total = int(
                len(kaggle_df)
            )


            # ----------------------------------------------
            # Sentiment counts
            # ----------------------------------------------

            kaggle_sentiment_counts = (

                kaggle_df["sentiment"]

                .value_counts()

            )


            kaggle_positive = int(

                kaggle_sentiment_counts.get(

                    "positive",

                    0

                )

            )


            kaggle_negative = int(

                kaggle_sentiment_counts.get(

                    "negative",

                    0

                )

            )


            kaggle_neutral = int(

                kaggle_sentiment_counts.get(

                    "neutral",

                    0

                )

            )


            # ----------------------------------------------
            # Average tweet length
            # ----------------------------------------------

            kaggle_average_length = float(

                round(

                    kaggle_df[
                        "tweet_length"
                    ].mean(),

                    2

                )

            )


            # ----------------------------------------------
            # Average length by sentiment
            # ----------------------------------------------

            kaggle_length_by_sentiment = (

                kaggle_df

                .groupby(
                    "sentiment"
                )[

                    "tweet_length"

                ]

                .mean()

                .round(2)

                .to_dict()

            )


            kaggle_positive_length = float(

                kaggle_length_by_sentiment.get(

                    "positive",

                    0

                )

            )


            kaggle_negative_length = float(

                kaggle_length_by_sentiment.get(

                    "negative",

                    0

                )

            )


            kaggle_neutral_length = float(

                kaggle_length_by_sentiment.get(

                    "neutral",

                    0

                )

            )


        # ==================================================
        # USER PREDICTION ANALYTICS
        # ==================================================

        conn = get_db_connection()

        cur = conn.cursor()


        # --------------------------------------------------
        # User prediction count
        # --------------------------------------------------

        cur.execute(
            """

            SELECT COUNT(*)

            FROM predictions

            """
        )


        user_total_result = cur.fetchone()


        user_total = (

            user_total_result[0]

            if user_total_result
            and user_total_result[0] is not None

            else 0

        )


        # --------------------------------------------------
        # User sentiment counts
        # --------------------------------------------------

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


        user_sentiment_row = cur.fetchone()


        user_positive = (

            user_sentiment_row[0]

            if user_sentiment_row
            and user_sentiment_row[0] is not None

            else 0

        )


        user_negative = (

            user_sentiment_row[1]

            if user_sentiment_row
            and user_sentiment_row[1] is not None

            else 0

        )


        user_neutral = (

            user_sentiment_row[2]

            if user_sentiment_row
            and user_sentiment_row[2] is not None

            else 0

        )


        # ==================================================
        # USER AVERAGE TWEET LENGTH
        # ==================================================

        cur.execute(
            """

            SELECT

                COALESCE(

                    AVG(
                        LENGTH(tweet_text)
                    ),

                    0

                )

            FROM predictions

            """
        )


        user_average_result = cur.fetchone()


        user_average_length = (

            float(user_average_result[0])

            if user_average_result
            and user_average_result[0] is not None

            else 0

        )


        # ==================================================
        # USER AVERAGE LENGTH BY SENTIMENT
        # ==================================================

        cur.execute(
            """

            SELECT

                COALESCE(

                    AVG(
                        LENGTH(tweet_text)
                    ) FILTER (

                        WHERE LOWER(sentiment) = 'positive'

                    ),

                    0

                ) AS positive_length,


                COALESCE(

                    AVG(
                        LENGTH(tweet_text)
                    ) FILTER (

                        WHERE LOWER(sentiment) = 'negative'

                    ),

                    0

                ) AS negative_length,


                COALESCE(

                    AVG(
                        LENGTH(tweet_text)
                    ) FILTER (

                        WHERE LOWER(sentiment) = 'neutral'

                    ),

                    0

                ) AS neutral_length


            FROM predictions

            """
        )


        user_length_row = cur.fetchone()


        user_positive_length = (

            float(user_length_row[0])

            if user_length_row
            and user_length_row[0] is not None

            else 0

        )


        user_negative_length = (

            float(user_length_row[1])

            if user_length_row
            and user_length_row[1] is not None

            else 0

        )


        user_neutral_length = (

            float(user_length_row[2])

            if user_length_row
            and user_length_row[2] is not None

            else 0

        )


        # ==================================================
        # COMBINE KAGGLE + USER DATA
        # ==================================================

        total_tweets = (

            kaggle_total
            +
            user_total

        )


        positive = (

            kaggle_positive
            +
            user_positive

        )


        negative = (

            kaggle_negative
            +
            user_negative

        )


        neutral = (

            kaggle_neutral
            +
            user_neutral

        )


        # ==================================================
        # COMBINED SENTIMENT PERCENTAGES
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
        # COMBINED AVERAGE TWEET LENGTH
        # ==================================================
        #
        # Weighted average:
        #
        # (Kaggle total characters +
        #  User total characters)
        #
        # / Total tweets
        #
        # ==================================================

        kaggle_total_characters = (

            kaggle_average_length
            *
            kaggle_total

        )


        user_total_characters = (

            user_average_length
            *
            user_total

        )


        if total_tweets > 0:

            combined_average_length = round(

                (

                    kaggle_total_characters
                    +
                    user_total_characters

                )
                /
                total_tweets,

                2

            )

        else:

            combined_average_length = 0


        # ==================================================
        # COMBINED AVERAGE POSITIVE LENGTH
        # ==================================================

        total_positive_tweets = (

            kaggle_positive
            +
            user_positive

        )


        positive_characters = (

            kaggle_positive_length
            *
            kaggle_positive

            +

            user_positive_length
            *
            user_positive

        )


        if total_positive_tweets > 0:

            combined_positive_length = round(

                positive_characters
                /
                total_positive_tweets,

                2

            )

        else:

            combined_positive_length = 0


        # ==================================================
        # COMBINED AVERAGE NEGATIVE LENGTH
        # ==================================================

        total_negative_tweets = (

            kaggle_negative
            +
            user_negative

        )


        negative_characters = (

            kaggle_negative_length
            *
            kaggle_negative

            +

            user_negative_length
            *
            user_negative

        )


        if total_negative_tweets > 0:

            combined_negative_length = round(

                negative_characters
                /
                total_negative_tweets,

                2

            )

        else:

            combined_negative_length = 0


        # ==================================================
        # COMBINED AVERAGE NEUTRAL LENGTH
        # ==================================================

        total_neutral_tweets = (

            kaggle_neutral
            +
            user_neutral

        )


        neutral_characters = (

            kaggle_neutral_length
            *
            kaggle_neutral

            +

            user_neutral_length
            *
            user_neutral

        )


        if total_neutral_tweets > 0:

            combined_neutral_length = round(

                neutral_characters
                /
                total_neutral_tweets,

                2

            )

        else:

            combined_neutral_length = 0


        # ==================================================
        # RETURN COMBINED ANALYTICS
        # ==================================================

        return jsonify({

            "success": True,


            # ------------------------------------------------
            # Data sources
            # ------------------------------------------------

            "data_source":
                "Kaggle Dataset + User Submitted Tweets",


            "kaggle_tweets":
                int(kaggle_total),


            "user_tweets":
                int(user_total),


            # ------------------------------------------------
            # Combined totals
            # ------------------------------------------------

            "total_tweets":
                int(total_tweets),


            "positive":
                int(positive),


            "negative":
                int(negative),


            "neutral":
                int(neutral),


            # ------------------------------------------------
            # Combined percentages
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Combined tweet lengths
            # ------------------------------------------------

            "average_tweet_length":
                float(
                    combined_average_length
                ),


            "average_length_positive":
                float(
                    combined_positive_length
                ),


            "average_length_negative":
                float(
                    combined_negative_length
                ),


            "average_length_neutral":
                float(
                    combined_neutral_length
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
    # These metrics were calculated using the test dataset.
    # They are NOT affected by user predictions.
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
