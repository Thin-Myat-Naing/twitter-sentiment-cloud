from flask import Flask, render_template, request, jsonify
import joblib


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
# LOAD MODEL
# ==========================================================

print("Loading sentiment model...")

model = joblib.load(
    MODEL_PATH
)

print("Sentiment model loaded.")


# ==========================================================
# LOAD TF-IDF VECTORIZER
# ==========================================================

print("Loading TF-IDF vectorizer...")

vectorizer = joblib.load(
    VECTORIZER_PATH
)

print("TF-IDF vectorizer loaded.")

print("All models loaded successfully!")


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================================
# PREDICTION API
# ==========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # Get JSON data
        data = request.get_json()

        # Get tweet text
        text = data.get(
            "text",
            ""
        ).strip()


        # Validate input
        if not text:

            return jsonify({
                "success": False,
                "error": "Please enter a tweet."
            }), 400


        # ==================================================
        # CLEAN TEXT
        # ==================================================

        import re

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

        # Keep hashtag words
        cleaned_text = re.sub(
            r"#(\w+)",
            r"\1",
            cleaned_text
        )

        # Convert contractions
        cleaned_text = cleaned_text.replace(
            "can't",
            "cannot"
        )

        cleaned_text = cleaned_text.replace(
            "won't",
            "will not"
        )

        cleaned_text = re.sub(
            r"n't\b",
            " not",
            cleaned_text
        )

        # Remove special characters
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
        # TF-IDF TRANSFORMATION
        # ==================================================

        features = vectorizer.transform(
            [cleaned_text]
        )


        # ==================================================
        # PREDICTION
        # ==================================================

        prediction = model.predict(
            features
        )[0]


        # ==================================================
        # PROBABILITY
        # ==================================================

        probabilities = model.predict_proba(
            features
        )[0]

        confidence = max(
            probabilities
        )


        # ==================================================
        # SENTIMENT
        # ==================================================

        sentiment = str(
            prediction
        ).capitalize()


        # ==================================================
        # RETURN RESULT
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
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )