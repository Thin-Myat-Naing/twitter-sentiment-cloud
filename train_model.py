import os
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ==========================================================
# 1. SETTINGS
# ==========================================================

DATA_PATH = "data/Tweets.csv"

MODEL_DIR = "models"

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sentiment_model.pkl"
)


# ==========================================================
# 2. CREATE MODEL DIRECTORY
# ==========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ==========================================================
# 3. LOAD DATASET
# ==========================================================

print("=" * 60)
print("PYTHON TWITTER SENTIMENT MODEL")
print("=" * 60)

print("\nLoading Twitter dataset...")

df = pd.read_csv(
    DATA_PATH
)

print("Dataset loaded successfully.")

print("\nColumns:")

print(
    df.columns.tolist()
)

print("\nNumber of rows:")

print(
    len(df)
)


# ==========================================================
# 4. SELECT REQUIRED COLUMNS
# ==========================================================

df = df[
    [
        "text",
        "sentiment"
    ]
]


# ==========================================================
# 5. REMOVE MISSING VALUES
# ==========================================================

print("\nRemoving missing values...")

df = df.dropna(
    subset=[
        "text",
        "sentiment"
    ]
)

df = df[
    df["text"].str.strip() != ""
]

print(
    "Rows after cleaning:",
    len(df)
)


# ==========================================================
# 6. CHECK SENTIMENT DISTRIBUTION
# ==========================================================

print("\nSentiment distribution:")

print(
    df["sentiment"].value_counts()
)


# ==========================================================
# 7. TEXT CLEANING FUNCTION
# ==========================================================

def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove Twitter mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Keep hashtag words but remove #
    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )

    # Convert contractions
    text = text.replace(
        "can't",
        "cannot"
    )

    text = text.replace(
        "won't",
        "will not"
    )

    text = re.sub(
        r"n't\b",
        " not",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================================
# 8. CLEAN TWEETS
# ==========================================================

print("\nCleaning tweet text...")

df["cleaned_text"] = df[
    "text"
].apply(
    clean_text
)


print("\nExample cleaned tweets:")

print(
    df[
        [
            "text",
            "cleaned_text",
            "sentiment"
        ]
    ].head(10).to_string(
        index=False
    )
)


# ==========================================================
# 9. REMOVE EMPTY TEXT
# ==========================================================

df = df[
    df["cleaned_text"].str.strip() != ""
]


# ==========================================================
# 10. FEATURES AND TARGET
# ==========================================================

X = df[
    "cleaned_text"
]

y = df[
    "sentiment"
]


# ==========================================================
# 11. TRAIN / TEST SPLIT
# ==========================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ==========================================================
# 12. TF-IDF VECTORIZER
# ==========================================================

print("\nCreating TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)


# ==========================================================
# 13. FIT TF-IDF
# ==========================================================

print("Training TF-IDF...")

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    "Training feature shape:",
    X_train_tfidf.shape
)

print(
    "Testing feature shape:",
    X_test_tfidf.shape
)


# ==========================================================
# 14. LOGISTIC REGRESSION
# ==========================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=500,
    C=2.0,
    class_weight="balanced"
)


model.fit(
    X_train_tfidf,
    y_train
)


print(
    "Model training completed."
)


# ==========================================================
# 15. PREDICTIONS
# ==========================================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test_tfidf
)


# ==========================================================
# 16. EVALUATION
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ==========================================================
# 17. DISPLAY RESULTS
# ==========================================================

print("\n")
print("=" * 60)
print("PYTHON MODEL EVALUATION RESULTS")
print("=" * 60)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)

print("=" * 60)


# ==========================================================
# 18. CLASSIFICATION REPORT
# ==========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================================
# 19. SAVE TF-IDF VECTORIZER
# ==========================================================

print("\nSaving TF-IDF vectorizer...")

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

print(
    "Saved:",
    VECTORIZER_PATH
)


# ==========================================================
# 20. SAVE MODEL
# ==========================================================

print("\nSaving sentiment model...")

joblib.dump(
    model,
    MODEL_PATH
)

print(
    "Saved:",
    MODEL_PATH
)


# ==========================================================
# 21. SHOW MODEL LABELS
# ==========================================================

print("\nModel sentiment classes:")

print(
    model.classes_
)


# ==========================================================
# 22. FINISHED
# ==========================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nFiles created:")

print(
    MODEL_PATH
)

print(
    VECTORIZER_PATH
)

print("\nYou can now run app.py.")