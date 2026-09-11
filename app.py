from flask import Flask, render_template, request, jsonify
import os
import re
import joblib
import pandas as pd
import numpy as np

from db.db import get_db_connection, setup_database


# ==========================================================
# FLASK APPLICATION
# ==========================================================

app = Flask(__name__)


# ==========================================================
# BASE DIRECTORY
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================================
# FILE PATHS
# ==========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "sentiment_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "tweet.csv"
)


# ==========================================================
# LOAD MODEL
# ==========================================================

try:

    model = joblib.load(MODEL_PATH)

    print("Sentiment model loaded successfully.")

except Exception as e:

    print("Error loading sentiment model:")
    print(e)

    model = None


# ==========================================================
# LOAD TF-IDF VECTORIZER
# ==========================================================

try:

    vectorizer = joblib.load(VECTORIZER_PATH)

    print("TF-IDF vectorizer loaded successfully.")

except Exception as e:

    print("Error loading TF-IDF vectorizer:")
    print(e)

    vectorizer = None


# ==========================================================
# LOAD KAGGLE DATASET
# ==========================================================

try:

    kaggle_df = pd.read_csv(DATASET_PATH)

    print(
        f"Kaggle dataset loaded successfully: "
        f"{len(kaggle_df)} rows"
    )

    print("Columns:")
    print(kaggle_df.columns.tolist())

    if "sentiment" in kaggle_df.columns:

        print("\nSentiment values:")

        print(
            kaggle_df["sentiment"]
            .value_counts(dropna=False)
        )

except Exception as e:

    print("Error loading tweet.csv:")
    print(e)

    kaggle_df = pd.DataFrame()


# ==========================================================
# DATABASE SETUP
# ==========================================================

try:

    setup_database()

except Exception as e:

    print("Database setup error:")
    print(e)


# ==========================================================
# SENTIMENT NORMALIZATION
# ==========================================================

def normalize_sentiment(value):

    if value is None:
        return None

    value = str(value).strip().lower()

    numeric_mapping = {

        "0": "neutral",

        "1": "positive",

        "2": "negative"

    }

    if value in numeric_mapping:

        return numeric_mapping[value]

    if value in ["pos", "positive"]:

        return "positive"

    if value in ["neg", "negative"]:

        return "negative"

    if value in ["neu", "neutral"]:

        return "neutral"

    return value


# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    if text is None:

        return ""

    text = str(text)


    # ------------------------------------------------------
    # Remove URLs
    # ------------------------------------------------------

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )


    # ------------------------------------------------------
    # Remove @mentions
    # ------------------------------------------------------

    text = re.sub(
        r"@\w+",
        "",
        text
    )


    # ------------------------------------------------------
    # Keep hashtag word but remove #
    # ------------------------------------------------------

    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )


    # ------------------------------------------------------
    # Common contractions
    # ------------------------------------------------------

    contractions = {

        "can't": "cannot",
        "won't": "will not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "wouldn't": "would not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "i'm": "i am",
        "you're": "you are",
        "we're": "we are",
        "they're": "they are",
        "it's": "it is",
        "that's": "that is"

    }


    for contraction, replacement in contractions.items():

        text = re.sub(
            rf"\b{re.escape(contraction)}\b",
            replacement,
            text,
            flags=re.IGNORECASE
        )


    # ------------------------------------------------------
    # Lowercase
    # ------------------------------------------------------

    text = text.lower()


    # ------------------------------------------------------
    # Keep English letters and basic punctuation
    # ------------------------------------------------------

    text = re.sub(
        r"[^a-z0-9\s!?.,']",
        " ",
        text
    )


    # ------------------------------------------------------
    # Remove extra spaces
    # ------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()


    return text


# ==========================================================
# PREPARE KAGGLE DATASET
# ==========================================================

def prepare_kaggle_dataset():

    global kaggle_df


    if kaggle_df.empty:

        return pd.DataFrame()


    df = kaggle_df.copy()


    # ------------------------------------------------------
    # Standardize column names
    # ------------------------------------------------------

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]


    # ------------------------------------------------------
    # Find ID column
    # ------------------------------------------------------

    id_col = None

    for candidate in [
        "textid",
        "id",
        "text_id",
        "tweet_id",
        "tweetid"
    ]:

        if candidate in df.columns:

            id_col = candidate

            break


    # ------------------------------------------------------
    # Create ID if necessary
    # ------------------------------------------------------

    if not id_col:

        df["textid"] = df.index

        id_col = "textid"


    # ------------------------------------------------------
    # Check required columns
    # ------------------------------------------------------

    if (
        "text" not in df.columns
        or
        "sentiment" not in df.columns
    ):

        print(
            "Missing required 'text' or "
            "'sentiment' column in tweet.csv"
        )

        return pd.DataFrame()


    # ------------------------------------------------------
    # Remove missing values
    # ------------------------------------------------------

    df = df.dropna(
        subset=[
            "text",
            "sentiment"
        ]
    )


    # ------------------------------------------------------
    # Remove duplicate IDs
    # ------------------------------------------------------

    df = df.drop_duplicates(
        subset=[id_col]
    )


    # ------------------------------------------------------
    # Normalize sentiment
    # ------------------------------------------------------

    df["sentiment"] = (
        df["sentiment"]
        .apply(normalize_sentiment)
    )


    # ------------------------------------------------------
    # Keep valid sentiment classes
    # ------------------------------------------------------

    df = df[
        df["sentiment"].isin(
            [
                "positive",
                "negative",
                "neutral"
            ]
        )
    ]


    # ------------------------------------------------------
    # Clean text
    # ------------------------------------------------------

    df["cleaned_text"] = (
        df["text"]
        .apply(clean_text)
    )


    # ------------------------------------------------------
    # Tweet length
    # ------------------------------------------------------

    df["tweet_length"] = (
        df["text"]
        .astype(str)
        .str.len()
    )


    return df


# ==========================================================
# PREPARE DATASET
# ==========================================================

kaggle_data = prepare_kaggle_dataset()


# ==========================================================
# HOME / ANALYZE PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================================
# DASHBOARD PAGE
# ==========================================================

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# ==========================================================
# PREDICT SENTIMENT
# ==========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()


        # --------------------------------------------------
        # Validate JSON
        # --------------------------------------------------

        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No JSON data received."

            }), 400


        # --------------------------------------------------
        # Get tweet text
        # --------------------------------------------------

        text = data.get(
            "text",
            ""
        ).strip()


        # --------------------------------------------------
        # Validate text
        # --------------------------------------------------

        if not text:

            return jsonify({

                "success": False,

                "error":
                    "Please enter a tweet."

            }), 400


        # --------------------------------------------------
        # Check model
        # --------------------------------------------------

        if (
            model is None
            or
            vectorizer is None
        ):

            return jsonify({

                "success": False,

                "error":
                    "Model or vectorizer "
                    "could not be loaded."

            }), 500


        # --------------------------------------------------
        # Clean text
        # --------------------------------------------------

        cleaned = clean_text(text)


        # --------------------------------------------------
        # TF-IDF transformation
        # --------------------------------------------------

        features = vectorizer.transform(
            [cleaned]
        )


        # --------------------------------------------------
        # Model prediction
        # --------------------------------------------------

        prediction = model.predict(
            features
        )[0]


        sentiment = normalize_sentiment(
            prediction
        )


        # --------------------------------------------------
        # Confidence
        # --------------------------------------------------

        confidence = None


        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    features
                )[0]
            )


            confidence = float(
                np.max(probabilities) * 100
            )


            confidence = round(
                confidence,
                2
            )


        # --------------------------------------------------
        # Save prediction to PostgreSQL
        # --------------------------------------------------

        conn = get_db_connection()

        cursor = conn.cursor()


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
        RETURNING id, created_at
        """


        cursor.execute(
            insert_query,
            (
                text,
                sentiment,
                confidence
            )
        )


        result = cursor.fetchone()


        prediction_id = result[0]

        created_at = result[1]


        conn.commit()


        cursor.close()

        conn.close()


        # --------------------------------------------------
        # Return prediction
        # --------------------------------------------------

        return jsonify({

            "success": True,

            "prediction":
                sentiment.capitalize(),

            "sentiment":
                sentiment,

            "confidence":
                confidence,

            "id":
                prediction_id,

            "created_at":
                created_at.isoformat()
                if created_at
                else None

        })


    except Exception as e:

        print(
            "Prediction error:",
            e
        )


        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ==========================================================
# PREDICTION HISTORY
# ==========================================================

@app.route("/api/history")
def history():

    try:

        conn = get_db_connection()

        cursor = conn.cursor()


        query = """
        SELECT
            id,
            tweet_text,
            sentiment,
            confidence,
            created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT 50
        """


        cursor.execute(query)

        rows = cursor.fetchall()


        cursor.close()

        conn.close()


        history_data = []


        for row in rows:

            prediction_id = row[0]

            tweet_text = row[1]

            sentiment = row[2]

            confidence = row[3]

            created_at = row[4]


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            if confidence is not None:

                confidence = float(
                    confidence
                )


            # ------------------------------------------------
            # Date
            # ------------------------------------------------

            if created_at:

                created_at = created_at.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )


            history_data.append({

                "id":
                    prediction_id,

                "tweet":
                    tweet_text,

                "sentiment":
                    sentiment.capitalize()
                    if sentiment
                    else "",

                "confidence":
                    confidence,

                "created_at":
                    created_at

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


# ==========================================================
# ANALYTICS API
# ==========================================================

@app.route("/api/analytics")
def analytics():

    try:

        # ==================================================
        # KAGGLE DATA
        # ==================================================

        if kaggle_data.empty:

            kaggle_total = 0

            kaggle_positive = 0

            kaggle_negative = 0

            kaggle_neutral = 0

            kaggle_avg_length = 0

            kaggle_avg_positive = 0

            kaggle_avg_negative = 0

            kaggle_avg_neutral = 0

        else:

            kaggle_total = len(
                kaggle_data
            )


            kaggle_positive = int(
                (
                    kaggle_data["sentiment"]
                    == "positive"
                ).sum()
            )


            kaggle_negative = int(
                (
                    kaggle_data["sentiment"]
                    == "negative"
                ).sum()
            )


            kaggle_neutral = int(
                (
                    kaggle_data["sentiment"]
                    == "neutral"
                ).sum()
            )


            # ------------------------------------------------
            # Kaggle average tweet length
            # ------------------------------------------------

            kaggle_avg_length = round(
                kaggle_data[
                    "tweet_length"
                ].mean(),
                2
            )


            # ------------------------------------------------
            # Positive lengths
            # ------------------------------------------------

            positive_lengths = (
                kaggle_data.loc[
                    kaggle_data["sentiment"]
                    == "positive",
                    "tweet_length"
                ]
            )


            # ------------------------------------------------
            # Negative lengths
            # ------------------------------------------------

            negative_lengths = (
                kaggle_data.loc[
                    kaggle_data["sentiment"]
                    == "negative",
                    "tweet_length"
                ]
            )


            # ------------------------------------------------
            # Neutral lengths
            # ------------------------------------------------

            neutral_lengths = (
                kaggle_data.loc[
                    kaggle_data["sentiment"]
                    == "neutral",
                    "tweet_length"
                ]
            )


            kaggle_avg_positive = round(
                positive_lengths.mean()
                if len(positive_lengths) > 0
                else 0,
                2
            )


            kaggle_avg_negative = round(
                negative_lengths.mean()
                if len(negative_lengths) > 0
                else 0,
                2
            )


            kaggle_avg_neutral = round(
                neutral_lengths.mean()
                if len(neutral_lengths) > 0
                else 0,
                2
            )


        # ==================================================
        # KAGGLE PERCENTAGES
        # ==================================================

        if kaggle_total > 0:

            kaggle_positive_percentage = round(
                (
                    kaggle_positive
                    / kaggle_total
                ) * 100,
                2
            )


            kaggle_negative_percentage = round(
                (
                    kaggle_negative
                    / kaggle_total
                ) * 100,
                2
            )


            kaggle_neutral_percentage = round(
                (
                    kaggle_neutral
                    / kaggle_total
                ) * 100,
                2
            )

        else:

            kaggle_positive_percentage = 0

            kaggle_negative_percentage = 0

            kaggle_neutral_percentage = 0


        # ==================================================
        # USER PREDICTION DATA
        # ==================================================

        conn = get_db_connection()

        cursor = conn.cursor()


        user_query = """
        SELECT
            sentiment,
            tweet_text
        FROM predictions
        """


        cursor.execute(
            user_query
        )


        user_rows = cursor.fetchall()


        cursor.close()

        conn.close()


        # ==================================================
        # USER COUNTS
        # ==================================================

        user_total = len(
            user_rows
        )


        user_positive = 0

        user_negative = 0

        user_neutral = 0


        user_tweet_lengths = []

        user_positive_lengths = []

        user_negative_lengths = []

        user_neutral_lengths = []


        for row in user_rows:

            predicted = normalize_sentiment(
                row[0]
            )


            tweet_text = row[1] or ""


            tweet_length = len(
                tweet_text
            )


            user_tweet_lengths.append(
                tweet_length
            )


            if predicted == "positive":

                user_positive += 1

                user_positive_lengths.append(
                    tweet_length
                )


            elif predicted == "negative":

                user_negative += 1

                user_negative_lengths.append(
                    tweet_length
                )


            elif predicted == "neutral":

                user_neutral += 1

                user_neutral_lengths.append(
                    tweet_length
                )


        # ==================================================
        # USER PERCENTAGES
        # ==================================================

        if user_total > 0:

            user_positive_percentage = round(
                (
                    user_positive
                    / user_total
                ) * 100,
                2
            )


            user_negative_percentage = round(
                (
                    user_negative
                    / user_total
                ) * 100,
                2
            )


            user_neutral_percentage = round(
                (
                    user_neutral
                    / user_total
                ) * 100,
                2
            )

        else:

            user_positive_percentage = 0

            user_negative_percentage = 0

            user_neutral_percentage = 0


        # ==================================================
        # USER AVERAGE LENGTH
        # ==================================================

        user_avg_length = round(
            np.mean(
                user_tweet_lengths
            ),
            2
        ) if user_tweet_lengths else 0


        user_avg_positive = round(
            np.mean(
                user_positive_lengths
            ),
            2
        ) if user_positive_lengths else 0


        user_avg_negative = round(
            np.mean(
                user_negative_lengths
            ),
            2
        ) if user_negative_lengths else 0


        user_avg_neutral = round(
            np.mean(
                user_neutral_lengths
            ),
            2
        ) if user_neutral_lengths else 0


        # ==================================================
        # COMBINED DATA
        # ==================================================

        combined_total = (
            kaggle_total
            + user_total
        )


        combined_positive = (
            kaggle_positive
            + user_positive
        )


        combined_negative = (
            kaggle_negative
            + user_negative
        )


        combined_neutral = (
            kaggle_neutral
            + user_neutral
        )


        # ==================================================
        # COMBINED PERCENTAGES
        # ==================================================

        if combined_total > 0:

            positive_percentage = round(
                (
                    combined_positive
                    / combined_total
                ) * 100,
                2
            )


            negative_percentage = round(
                (
                    combined_negative
                    / combined_total
                ) * 100,
                2
            )


            neutral_percentage = round(
                (
                    combined_neutral
                    / combined_total
                ) * 100,
                2
            )

        else:

            positive_percentage = 0

            negative_percentage = 0

            neutral_percentage = 0


        # ==================================================
        # COMBINED AVERAGE LENGTH
        # ==================================================

        combined_lengths = []


        if not kaggle_data.empty:

            combined_lengths.extend(
                kaggle_data[
                    "tweet_length"
                ].tolist()
            )


        combined_lengths.extend(
            user_tweet_lengths
        )


        combined_average_length = round(
            np.mean(
                combined_lengths
            ),
            2
        ) if combined_lengths else 0


        # ==================================================
        # COMBINED SENTIMENT AVERAGE LENGTHS
        # ==================================================

        combined_positive_lengths = []

        combined_negative_lengths = []

        combined_neutral_lengths = []


        # Kaggle positive

        if not kaggle_data.empty:

            combined_positive_lengths.extend(
                kaggle_data.loc[
                    kaggle_data["sentiment"]
                    == "positive",
                    "tweet_length"
                ].tolist()
            )


            combined_negative_lengths.extend(
                kaggle_data.loc[
                    kaggle_data["sentiment"]
                    == "negative",
                    "tweet_length"
                ].tolist()
            )


            combined_neutral_lengths.extend(
                kaggle_data.loc[
                    kaggle_data["sentiment"]
                    == "neutral",
                    "tweet_length"
                ].tolist()
            )


        # User positive

        combined_positive_lengths.extend(
            user_positive_lengths
        )


        # User negative

        combined_negative_lengths.extend(
            user_negative_lengths
        )


        # User neutral

        combined_neutral_lengths.extend(
            user_neutral_lengths
        )


        # --------------------------------------------------
        # Final average lengths
        # --------------------------------------------------

        combined_avg_positive = round(
            np.mean(
                combined_positive_lengths
            ),
            2
        ) if combined_positive_lengths else 0


        combined_avg_negative = round(
            np.mean(
                combined_negative_lengths
            ),
            2
        ) if combined_negative_lengths else 0


        combined_avg_neutral = round(
            np.mean(
                combined_neutral_lengths
            ),
            2
        ) if combined_neutral_lengths else 0


        # ==================================================
        # RETURN ANALYTICS
        # ==================================================

        return jsonify({

            "success": True,


            # ----------------------------------------------
            # KAGGLE
            # ----------------------------------------------

            "kaggle_tweets":
                kaggle_total,

            "kaggle_positive":
                kaggle_positive,

            "kaggle_negative":
                kaggle_negative,

            "kaggle_neutral":
                kaggle_neutral,

            "kaggle_positive_percentage":
                kaggle_positive_percentage,

            "kaggle_negative_percentage":
                kaggle_negative_percentage,

            "kaggle_neutral_percentage":
                kaggle_neutral_percentage,


            # ----------------------------------------------
            # USER
            # ----------------------------------------------

            "user_tweets":
                user_total,

            "user_positive":
                user_positive,

            "user_negative":
                user_negative,

            "user_neutral":
                user_neutral,

            "user_positive_percentage":
                user_positive_percentage,

            "user_negative_percentage":
                user_negative_percentage,

            "user_neutral_percentage":
                user_neutral_percentage,


            # ----------------------------------------------
            # COMBINED
            # ----------------------------------------------

            "total_tweets":
                combined_total,

            "positive":
                combined_positive,

            "negative":
                combined_negative,

            "neutral":
                combined_neutral,

            "positive_percentage":
                positive_percentage,

            "negative_percentage":
                negative_percentage,

            "neutral_percentage":
                neutral_percentage,


            # ----------------------------------------------
            # AVERAGE LENGTH
            # ----------------------------------------------

            "average_tweet_length":
                combined_average_length,

            "average_length_positive":
                combined_avg_positive,

            "average_length_negative":
                combined_avg_negative,

            "average_length_neutral":
                combined_avg_neutral

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


# ==========================================================
# RUN APPLICATION
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