import os

from flask import Flask, render_template, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.functions import col

# ==========================================================
# 1. CREATE FLASK APP
# ==========================================================

app = Flask(__name__)


# ==========================================================
# 2. PROJECT PATH
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PREPROCESSING_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "twitter_preprocessing"
)

CLASSIFICATION_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "twitter_logistic_regression"
)


# ==========================================================
# 3. CREATE SPARK SESSION
# ==========================================================

spark = SparkSession.builder \
    .appName("TwitterSentimentWebApplication") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# ==========================================================
# 4. LOAD TRAINED MODELS
# ==========================================================

print("Loading preprocessing model...")

preprocessing_model = PipelineModel.load(
    PREPROCESSING_MODEL_PATH
)

print("Preprocessing model loaded.")


print("Loading classification model...")

classification_model = LogisticRegressionModel.load(
    CLASSIFICATION_MODEL_PATH
)

print("Classification model loaded.")

print("All models loaded successfully!")


# ==========================================================
# 5. HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================================
# 6. PREDICTION
# ==========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # Get data from website
        data = request.get_json()

        # Get tweet
        text = data.get(
            "text",
            ""
        ).strip()

        # Check empty input
        if not text:

            return jsonify({
                "success": False,
                "error": "Please enter a tweet."
            }), 400


        # ==================================================
        # CREATE SPARK DATAFRAME
        # ==================================================

        input_df = spark.createDataFrame(
            [(text,)],
            ["text"]
        )


        # ==================================================
        # CREATE cleaned_text COLUMN
        # ==================================================

        input_df = input_df.withColumn(
            "cleaned_text",
            col("text")
        )


        # ==================================================
        # PREPROCESS TEXT
        # ==================================================

        input_processed = (
            preprocessing_model
            .transform(input_df)
        )


        # ==================================================
        # MAKE PREDICTION
        # ==================================================

        prediction = (
            classification_model
            .transform(input_processed)
        )


        # ==================================================
        # GET RESULT
        # ==================================================

        result = prediction.select(
            "prediction",
            "probability"
        ).collect()[0]


        predicted_label = int(
            result["prediction"]
        )


        # ==================================================
        # CONFIDENCE
        # ==================================================

        probabilities = result["probability"]

        confidence = float(
            max(probabilities)
        )


        # ==================================================
        # SENTIMENT LABEL
        # ==================================================

        label_mapping = {
            0: "Neutral",
            1: "Positive",
            2: "Negative"
        }
        sentiment = label_mapping.get(
            predicted_label,
            "Unknown"
        )


        # ==================================================
        # RETURN RESULT TO WEBSITE
        # ==================================================

        return jsonify({

            "success": True,

            "text": text,

            "sentiment": sentiment,

            "confidence": round(
                confidence * 100,
                2
            )

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
# 7. START FLASK SERVER
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