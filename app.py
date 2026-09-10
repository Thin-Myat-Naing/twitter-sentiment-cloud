from flask import Flask, render_template, request, jsonify
import os
import re
import joblib
import pandas as pd
import numpy as np
import psycopg2

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from db.db import get_db_connection, setup_database

df = pd.read_csv("tweet.csv")

print("Columns:")
print(df.columns.tolist())

print("\nNumber of rows:")
print(len(df))

print("\nSentiment values:")
if "sentiment" in df.columns:
    print(df["sentiment"].value_counts(dropna=False))

print("\nFirst 10 rows:")
if "text" in df.columns and "sentiment" in df.columns:
    print(df[["text", "sentiment"]].head(10).to_string())

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

    # Numeric labels used by many Kaggle datasets
    numeric_mapping = {

        "0": "neutral",

        "1": "positive",

        "2": "negative"

    }

    if value in numeric_mapping:

        return numeric_mapping[value]

    # Common alternative names
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

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Keep hashtag word but remove #
    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )

    # Common contractions
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

    # Lowercase
    text = text.lower()

    # Keep English letters and basic punctuation
    text = re.sub(
        r"[^a-z0-9\s!?.,']",
        " ",
        text
    )

    # Remove extra spaces
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

    # Standardize column names to lowercase
    df.columns = [str(col).strip().lower() for col in df.columns]

    # Map potential ID column names
    id_col = None
    for candidate in ["textid", "id", "text_id", "tweet_id", "tweetid"]:
        if candidate in df.columns:
            id_col = candidate
            break

    # If no ID column exists, create one from index
    if not id_col:
        df["textid"] = df.index
        id_col = "textid"

    # Ensure required text and sentiment columns exist
    if "text" not in df.columns or "sentiment" not in df.columns:
        print("Missing required 'text' or 'sentiment' column in tweet.csv")
        return pd.DataFrame()

    # Remove missing text/sentiment
    df = df.dropna(
        subset=[
            "text",
            "sentiment"
        ]
    )

    # Remove duplicate text IDs
    df = df.drop_duplicates(
        subset=[id_col]
    )

    # Normalize sentiment
    df["sentiment"] = (
        df["sentiment"]
        .apply(normalize_sentiment)
    )

    # Keep only target classes
    df = df[
        df["sentiment"].isin(
            [
                "positive",
                "negative",
                "neutral"
            ]
        )
    ]

    # Clean text
    df["cleaned_text"] = (
        df["text"]
        .apply(clean_text)
    )

    # Tweet length
    df["tweet_length"] = (
        df["text"]
        .astype(str)
        .str.len()
    )

    return df


kaggle_data = prepare_kaggle_dataset()


# ==========================================================
# CALCULATE MODEL METRICS
# ==========================================================

def calculate_metrics(y_true, y_pred):

    if len(y_true) == 0:

        return {

            "accuracy": None,

            "precision": None,

            "recall": None,

            "f1_score": None

        }

    y_true = [
        normalize_sentiment(x)
        for x in y_true
    ]

    y_pred = [
        normalize_sentiment(x)
        for x in y_pred
    ]

    return {

        "accuracy": round(
            accuracy_score(
                y_true,
                y_pred
            ) * 100,
            2
        ),

        "precision": round(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ) * 100,
            2
        ),

        "recall": round(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ) * 100,
            2
        ),

        "f1_score": round(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ) * 100,
            2
        )

    }


# ==========================================================
# KAGGLE MODEL PERFORMANCE
# ==========================================================

def calculate_kaggle_performance():

    if kaggle_data.empty:

        return {

            "accuracy": None,

            "precision": None,

            "recall": None,

            "f1_score": None

        }

    if model is None or vectorizer is None:

        return {

            "accuracy": None,

            "precision": None,

            "recall": None,

            "f1_score": None

        }

    try:

        X = vectorizer.transform(
            kaggle_data["cleaned_text"]
        )

        predictions = model.predict(X)

        predictions = [
            normalize_sentiment(x)
            for x in predictions
        ]

        actual = (
            kaggle_data["sentiment"]
            .tolist()
        )

        return calculate_metrics(
            actual,
            predictions
        )

    except Exception as e:

        print(
            "Kaggle performance error:",
            e
        )

        return {

            "accuracy": None,

            "precision": None,

            "recall": None,

            "f1_score": None

        }


# Calculate performance metrics after dataset is prepared
KAGGLE_PERFORMANCE = (
    calculate_kaggle_performance()
)


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

        if not data:

            return jsonify({

                "success": False,

                "error": "No JSON data received."

            }), 400


        text = data.get(
            "text",
            ""
        ).strip()


        # ==================================================
        # ACTUAL SENTIMENT
        # ==================================================

        actual_sentiment = data.get(
            "actual_sentiment"
        )


        if actual_sentiment:

            actual_sentiment = normalize_sentiment(
                actual_sentiment
            )

            if actual_sentiment not in [
                "positive",
                "negative",
                "neutral"
            ]:

                return jsonify({

                    "success": False,

                    "error":
                        "Actual sentiment must be "
                        "Positive, Negative, or Neutral."

                }), 400

        else:

            actual_sentiment = None


        # ==================================================
        # VALIDATE TEXT
        # ==================================================

        if not text:

            return jsonify({

                "success": False,

                "error": "Please enter a tweet."

            }), 400


        # ==================================================
        # CHECK MODEL
        # ==================================================

        if model is None or vectorizer is None:

            return jsonify({

                "success": False,

                "error":
                    "Model or vectorizer could not be loaded."

            }), 500


        # ==================================================
        # CLEAN TEXT
        # ==================================================

        cleaned = clean_text(text)


        # ==================================================
        # TF-IDF
        # ==================================================

        features = vectorizer.transform(
            [cleaned]
        )


        # ==================================================
        # PREDICTION
        # ==================================================

        prediction = model.predict(
            features
        )[0]


        sentiment = normalize_sentiment(
            prediction
        )


        # ==================================================
        # CONFIDENCE
        # ==================================================

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


        # ==================================================
        # SAVE TO POSTGRESQL
        # ==================================================

        conn = get_db_connection()

        cursor = conn.cursor()


        insert_query = """
        INSERT INTO predictions
        (
            tweet_text,
            sentiment,
            actual_sentiment,
            confidence
        )
        VALUES
        (
            %s,
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
                actual_sentiment,
                confidence
            )
        )


        result = cursor.fetchone()

        prediction_id = result[0]

        created_at = result[1]


        conn.commit()

        cursor.close()

        conn.close()


        # ==================================================
        # RESPONSE
        # ==================================================

        return jsonify({

            "success": True,

            "prediction": sentiment.capitalize(),

            "sentiment": sentiment,

            "confidence": confidence,

            "actual_sentiment":
                actual_sentiment,

            "id": prediction_id,

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

            "error": str(e)

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
            actual_sentiment,
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

            actual_sentiment = row[3]

            confidence = row[4]

            created_at = row[5]


            # Convert confidence to normal float
            if confidence is not None:

                confidence = float(
                    confidence
                )


            # Format date
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

                "actual_sentiment":
                    actual_sentiment.capitalize()
                    if actual_sentiment
                    else None,

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

            "error": str(e)

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


            kaggle_avg_length = round(
                kaggle_data[
                    "tweet_length"
                ].mean(),
                2
            )


            positive_lengths = (
                kaggle_data.loc[
                    kaggle_data["sentiment"]
                    == "positive",
                    "tweet_length"
                ]
            )


            negative_lengths = (
                kaggle_data.loc[
                    kaggle_data["sentiment"]
                    == "negative",
                    "tweet_length"
                ]
            )


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
        # USER DATA
        # ==================================================

        conn = get_db_connection()

        cursor = conn.cursor()


        user_query = """
        SELECT
            sentiment,
            actual_sentiment,
            tweet_text
        FROM predictions
        """


        cursor.execute(
            user_query
        )

        user_rows = cursor.fetchall()


        cursor.close()

        conn.close()


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

            tweet_text = row[2] or ""

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
        # RETURN DATA
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
                kaggle_avg_positive,

            "average_length_negative":
                kaggle_avg_negative,

            "average_length_neutral":
                kaggle_avg_neutral

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

    try:

        # ==================================================
        # KAGGLE PERFORMANCE
        # ==================================================

        kaggle_accuracy = (
            KAGGLE_PERFORMANCE["accuracy"]
        )

        kaggle_precision = (
            KAGGLE_PERFORMANCE["precision"]
        )

        kaggle_recall = (
            KAGGLE_PERFORMANCE["recall"]
        )

        kaggle_f1 = (
            KAGGLE_PERFORMANCE["f1_score"]
        )


        # ==================================================
        # USER PERFORMANCE
        # ==================================================

        conn = get_db_connection()

        cursor = conn.cursor()


        query = """
        SELECT
            sentiment,
            actual_sentiment
        FROM predictions
        WHERE actual_sentiment IS NOT NULL
        AND TRIM(actual_sentiment) <> ''
        """


        cursor.execute(query)

        rows = cursor.fetchall()


        cursor.close()

        conn.close()


        y_true = []

        y_pred = []


        for row in rows:

            predicted = normalize_sentiment(
                row[0]
            )

            actual = normalize_sentiment(
                row[1]
            )


            if (
                predicted in [
                    "positive",
                    "negative",
                    "neutral"
                ]
                and
                actual in [
                    "positive",
                    "negative",
                    "neutral"
                ]
            ):

                y_pred.append(
                    predicted
                )

                y_true.append(
                    actual
                )


        # ==================================================
        # CALCULATE USER METRICS
        # ==================================================

        if len(y_true) > 0:

            user_metrics = calculate_metrics(
                y_true,
                y_pred
            )

            user_accuracy = (
                user_metrics["accuracy"]
            )

            user_precision = (
                user_metrics["precision"]
            )

            user_recall = (
                user_metrics["recall"]
            )

            user_f1 = (
                user_metrics["f1_score"]
            )

        else:

            # IMPORTANT:
            # Do not show 0%.
            # There is no labelled user data yet.

            user_accuracy = None

            user_precision = None

            user_recall = None

            user_f1 = None


        # ==================================================
        # RETURN
        # ==================================================

        return jsonify({

            "success": True,


            # Kaggle
            "kaggle_accuracy":
                kaggle_accuracy,

            "kaggle_precision":
                kaggle_precision,

            "kaggle_recall":
                kaggle_recall,

            "kaggle_f1_score":
                kaggle_f1,


            # User
            "user_labeled_tweets":
                len(y_true),

            "user_accuracy":
                user_accuracy,

            "user_precision":
                user_precision,

            "user_recall":
                user_recall,

            "user_f1_score":
                user_f1

        })


    except Exception as e:

        print(
            "Model performance error:",
            e
        )

        return jsonify({

            "success": False,

            "error": str(e)

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