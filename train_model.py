from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lower,
    regexp_replace,
    trim
)

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    Tokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF
)

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import os
import json


# ==========================================================
# 1. SETTINGS
# ==========================================================

DATA_PATH = "data/Tweets.csv"

PREPROCESSING_MODEL_PATH = "models/twitter_preprocessing"
CLASSIFICATION_MODEL_PATH = "models/twitter_logistic_regression"

LABEL_MAPPING_PATH = "models/label_mapping.json"


# ==========================================================
# 2. CREATE SPARK SESSION
# ==========================================================

print("=" * 60)
print("TWITTER SENTIMENT ANALYSIS")
print("=" * 60)

print("\nStarting Spark...")

spark = SparkSession.builder \
    .appName("TwitterSentimentTraining") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("Spark started successfully.")


# ==========================================================
# 3. CREATE MODEL DIRECTORY
# ==========================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ==========================================================
# 4. LOAD DATASET
# ==========================================================

print("\nLoading Twitter dataset...")

df = spark.read.csv(
    DATA_PATH,
    header=True,
    inferSchema=True
)

print("Dataset loaded successfully.")

print("\nDataset columns:")

print(df.columns)

print("\nNumber of rows:")

print(df.count())


# ==========================================================
# 5. CHECK COLUMNS
# ==========================================================

if "text" not in df.columns:

    raise ValueError(
        "\nERROR: 'text' column was not found.\n"
        f"Available columns: {df.columns}"
    )


if "sentiment" not in df.columns:

    raise ValueError(
        "\nERROR: 'sentiment' column was not found.\n"
        f"Available columns: {df.columns}"
    )


# ==========================================================
# 6. SELECT REQUIRED COLUMNS
# ==========================================================

df = df.select(
    "text",
    "sentiment"
)


# ==========================================================
# 7. REMOVE NULL VALUES
# ==========================================================

print("\nRemoving missing values...")

df = df.dropna(
    subset=[
        "text",
        "sentiment"
    ]
)

df = df.filter(
    trim(col("text")) != ""
)

print(
    "Rows after cleaning:",
    df.count()
)


# ==========================================================
# 8. CHECK SENTIMENT DISTRIBUTION
# ==========================================================

print("\nSentiment distribution:")

df.groupBy(
    "sentiment"
).count().orderBy(
    col("count").desc()
).show()


# ==========================================================
# 9. TEXT CLEANING
# ==========================================================

print("\nCleaning Twitter text...")


# Convert to lowercase

df = df.withColumn(
    "cleaned_text",
    lower(col("text"))
)


# Remove URLs

df = df.withColumn(
    "cleaned_text",
    regexp_replace(
        col("cleaned_text"),
        r"https?://\S+|www\.\S+",
        " "
    )
)


# Remove Twitter mentions

df = df.withColumn(
    "cleaned_text",
    regexp_replace(
        col("cleaned_text"),
        r"@\w+",
        " "
    )
)


# Keep hashtag word but remove # symbol

df = df.withColumn(
    "cleaned_text",
    regexp_replace(
        col("cleaned_text"),
        r"#(\w+)",
        r"$1"
    )
)


# Convert contractions

df = df.withColumn(
    "cleaned_text",
    regexp_replace(
        col("cleaned_text"),
        r"can't",
        "cannot"
    )
)

df = df.withColumn(
    "cleaned_text",
    regexp_replace(
        col("cleaned_text"),
        r"won't",
        "will not"
    )
)

df = df.withColumn(
    "cleaned_text",
    regexp_replace(
        col("cleaned_text"),
        r"n't\b",
        " not"
    )
)


# Remove special characters

df = df.withColumn(
    "cleaned_text",
    regexp_replace(
        col("cleaned_text"),
        r"[^a-zA-Z\s]",
        " "
    )
)


# Remove extra spaces

df = df.withColumn(
    "cleaned_text",
    regexp_replace(
        col("cleaned_text"),
        r"\s+",
        " "
    )
)


# Trim

df = df.withColumn(
    "cleaned_text",
    trim(col("cleaned_text"))
)


# ==========================================================
# 10. SHOW CLEANED DATA
# ==========================================================

print("\nExample cleaned tweets:")

df.select(
    "text",
    "cleaned_text",
    "sentiment"
).show(
    10,
    truncate=False
)


# ==========================================================
# 11. ENCODE SENTIMENT
# ==========================================================

print("\nEncoding sentiment labels...")

label_indexer = StringIndexer(
    inputCol="sentiment",
    outputCol="label",
    handleInvalid="keep"
)

label_model = label_indexer.fit(df)

df = label_model.transform(df)


# ==========================================================
# 12. DISPLAY LABEL MAPPING
# ==========================================================

print("\nSentiment label mapping:")

labels = label_model.labels

label_mapping = {}

for index, label in enumerate(labels):

    label_mapping[index] = label

    print(
        f"{index} -> {label}"
    )


# ==========================================================
# 13. SAVE LABEL MAPPING
# ==========================================================

with open(
    LABEL_MAPPING_PATH,
    "w"
) as file:

    json.dump(
        label_mapping,
        file,
        indent=4
    )

print(
    "\nLabel mapping saved:"
)

print(
    LABEL_MAPPING_PATH
)


# ==========================================================
# 14. SPLIT DATASET
# ==========================================================

print("\nSplitting dataset...")

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print(
    "Training rows:",
    train_df.count()
)

print(
    "Testing rows:",
    test_df.count()
)


# ==========================================================
# 15. TOKENIZATION
# ==========================================================

tokenizer = Tokenizer(
    inputCol="cleaned_text",
    outputCol="words"
)


# ==========================================================
# 16. STOP WORD REMOVAL
# ==========================================================

stop_words = StopWordsRemover(
    inputCol="words",
    outputCol="filtered_words"
)


# ==========================================================
# 17. COUNT VECTORIZER
# ==========================================================

count_vectorizer = CountVectorizer(
    inputCol="filtered_words",
    outputCol="raw_features",
    vocabSize=20000,
    minDF=2.0
)


# ==========================================================
# 18. IDF
# ==========================================================

idf = IDF(
    inputCol="raw_features",
    outputCol="features"
)


# ==========================================================
# 19. PREPROCESSING PIPELINE
# ==========================================================

preprocessing_pipeline = Pipeline(
    stages=[
        tokenizer,
        stop_words,
        count_vectorizer,
        idf
    ]
)


# ==========================================================
# 20. TRAIN PREPROCESSING PIPELINE
# ==========================================================

print(
    "\nTraining preprocessing pipeline..."
)

preprocessing_model = preprocessing_pipeline.fit(
    train_df
)

print(
    "Preprocessing pipeline trained successfully."
)


# ==========================================================
# 21. SAVE PREPROCESSING MODEL
# ==========================================================

print(
    "\nSaving preprocessing model..."
)

preprocessing_model.write().overwrite().save(
    PREPROCESSING_MODEL_PATH
)

print(
    "Preprocessing model saved:"
)

print(
    PREPROCESSING_MODEL_PATH
)


# ==========================================================
# 22. TRANSFORM DATA
# ==========================================================

print(
    "\nCreating TF-IDF features..."
)

train_features = preprocessing_model.transform(
    train_df
)

test_features = preprocessing_model.transform(
    test_df
)


# ==========================================================
# 23. LOGISTIC REGRESSION
# ==========================================================

print(
    "\nCreating Logistic Regression..."
)

logistic_regression = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=200,
    regParam=0.05,
    elasticNetParam=0.0
)


# ==========================================================
# 24. TRAIN CLASSIFIER
# ==========================================================

print(
    "Training Logistic Regression..."
)

classification_model = logistic_regression.fit(
    train_features
)

print(
    "Logistic Regression training completed."
)


# ==========================================================
# 25. SAVE CLASSIFICATION MODEL
# ==========================================================

print(
    "\nSaving classification model..."
)

classification_model.write().overwrite().save(
    CLASSIFICATION_MODEL_PATH
)

print(
    "Classification model saved:"
)

print(
    CLASSIFICATION_MODEL_PATH
)


# ==========================================================
# 26. PREDICTIONS
# ==========================================================

print(
    "\nGenerating predictions..."
)

predictions = classification_model.transform(
    test_features
)


# ==========================================================
# 27. EVALUATION
# ==========================================================

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = accuracy_evaluator.evaluate(
    predictions
)


precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

precision = precision_evaluator.evaluate(
    predictions
)


recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)

recall = recall_evaluator.evaluate(
    predictions
)


f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

f1_score = f1_evaluator.evaluate(
    predictions
)


# ==========================================================
# 28. DISPLAY RESULTS
# ==========================================================

print("\n")
print("=" * 60)
print("IMPROVED MODEL EVALUATION RESULTS")
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
    f"F1 Score : {f1_score * 100:.2f}%"
)

print("=" * 60)


# ==========================================================
# 29. SAMPLE PREDICTIONS
# ==========================================================

print("\nSample predictions:")

predictions.select(
    "text",
    "sentiment",
    "label",
    "prediction"
).show(
    10,
    truncate=False
)


# ==========================================================
# 30. FINISH
# ==========================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nSaved models:")

print(
    "1.",
    PREPROCESSING_MODEL_PATH
)

print(
    "2.",
    CLASSIFICATION_MODEL_PATH
)

print(
    "3.",
    LABEL_MAPPING_PATH
)

print("\nYou can now run app.py.")


# ==========================================================
# 31. STOP SPARK
# ==========================================================

spark.stop()