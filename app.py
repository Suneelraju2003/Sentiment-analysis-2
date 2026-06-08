import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Customer Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ----------------------------
# NLTK
# ----------------------------

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ----------------------------
# Text Preprocessing
# ----------------------------

def preprocess(text):

    text = str(text).lower()

    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?]', '', text)
    text = re.sub(r'\d+', '', text)

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)

# ----------------------------
# Sidebar Navigation
# ----------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Model Performance",
        "Sentiment Prediction"
    ]
)

# =====================================================
# PAGE 1
# =====================================================

if page == "Model Performance":

    st.title("Customer Review Sentiment Analysis")

    st.header("Project Overview")

    st.write("""
    This project predicts customer sentiment from product reviews
    using TF-IDF Vectorization and Logistic Regression.
    """)

    st.divider()

    st.header("Model Performance")

    performance_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Value": [
            0.92,   # replace
            0.91,   # replace
            0.92,   # replace
            0.91    # replace
        ]
    })

    st.dataframe(
        performance_df,
        use_container_width=True
    )

    st.divider()

    st.header("Performance Visualization")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        performance_df["Metric"],
        performance_df["Value"]
    )

    ax.set_ylim([0,1])

    ax.set_ylabel("Score")

    st.pyplot(fig)

    st.divider()

    st.header("Training vs Validation Accuracy")

    comparison_df = pd.DataFrame({
        "Dataset":[
            "Training",
            "Validation"
        ],
        "Accuracy":[
            0.94,  # replace
            0.92   # replace
        ]
    })

    fig2, ax2 = plt.subplots(figsize=(6,4))

    ax2.bar(
        comparison_df["Dataset"],
        comparison_df["Accuracy"]
    )

    ax2.set_ylim([0,1])

    st.pyplot(fig2)

    st.divider()

    st.header("Confusion Matrix")

    confusion_df = pd.DataFrame(
        [
            [450,50],
            [40,460]
        ],
        columns=["Pred Negative","Pred Positive"],
        index=["Actual Negative","Actual Positive"]
    )

    st.dataframe(confusion_df)

# =====================================================
# PAGE 2
# =====================================================

elif page == "Sentiment Prediction":

    st.title("Customer Review Sentiment Prediction")

    st.write(
        "Enter a customer review below."
    )

    review = st.text_area(
        "Customer Review",
        height=200
    )

    if st.button("Predict Sentiment"):

        if review.strip() == "":
            st.warning("Please enter a review.")
        else:

            cleaned_review = preprocess(review)

            transformed_text = vectorizer.transform(
                [cleaned_review]
            )

            prediction = model.predict(
                transformed_text
            )[0]

            try:
                probability = np.max(
                    model.predict_proba(
                        transformed_text
                    )
                )
            except:
                probability = None

            st.subheader("Prediction")

            if str(prediction).lower() == "positive":
                st.success(
                    f"Positive Sentiment"
                )

            elif str(prediction).lower() == "negative":
                st.error(
                    f"Negative Sentiment"
                )

            else:
                st.info(
                    f"{prediction}"
                )

            if probability is not None:

                st.write(
                    f"Confidence Score: {probability:.2%}"
                )

            st.divider()

            st.subheader("Processed Text")

            st.write(cleaned_review)

    st.divider()

    st.subheader("Example Reviews")

    st.info(
        "This product exceeded my expectations. Excellent quality."
    )

    st.info(
        "Waste of money. Very poor performance."
    )
