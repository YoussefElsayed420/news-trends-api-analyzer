import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

NEGATIVE_KEYWORDS = {"killed", "fire", "disaster", "attack", "dead", "explosion", "crash", "injured", "shooting", "earthquake", "flood", "bomb", "collapse", "tragedy", "violence", "war", "death", "fatal", "victim", "casualty"}

def contains_negative_keywords(text):
    return any(word in text.lower() for word in NEGATIVE_KEYWORDS)

def get_sentiment_batch(texts, batch_size=16):
    sentiments = []
    confidences = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)
            for j, text in enumerate(batch):
                if contains_negative_keywords(text):
                    sentiments.append("Negative")
                    confidences.append(1.0)
                else:
                    sentiments.append(LABEL_MAP[labels[j]])
                    confidences.append(float(confs[j]))
    return sentiments, confidences

def main():
    input_path = "data/top_headlines.csv"
    if not os.path.exists(input_path):
        print(f"❌ ERROR: {input_path} not found. Please ensure the news fetch step runs successfully before sentiment analysis.")
        sys.exit(1)
    df = pd.read_csv(input_path)
    df.dropna(subset=["title"], inplace=True)

    sentiments, confidences = get_sentiment_batch(df["title"].tolist())
    df["sentiment"] = sentiments
    df["sentiment_confidence"] = confidences

    filtered_df = df[df["sentiment_confidence"] >= 0.7].copy()

    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    if filtered_df.empty:
        print("No headlines with confidence >= 0.7. No output will be generated.")
        return

    plt.figure(figsize=(6, 4))
    sns.countplot(data=filtered_df, x="sentiment", palette="coolwarm")
    plt.title("Sentiment of News Headlines (Confidence ≥ 0.7)")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Headlines")
    plt.tight_layout()
    plt.savefig("output/sentiment_chart.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(df["sentiment_confidence"], bins=20, color="skyblue", edgecolor="black")
    plt.title("Distribution of Sentiment Confidence Scores")
    plt.xlabel("Sentiment Confidence")
    plt.ylabel("Number of Headlines")
    plt.tight_layout()
    plt.savefig("output/sentiment_confidence_hist.png")
    plt.close()

    vectorizer = CountVectorizer(stop_words="english")
    if not filtered_df.empty:
        X = vectorizer.fit_transform(filtered_df["title"])
        word_counts = X.sum(axis=0).A1
        words = vectorizer.get_feature_names_out()
        top_words_df = pd.DataFrame({"word": words, "count": word_counts})
        top_words_df = top_words_df.sort_values(by="count", ascending=False).head(10)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=top_words_df, x="count", y="word", palette="Blues_d")
        plt.title("Top Keywords in News Headlines (Confidence ≥ 0.7)")
        plt.xlabel("Count")
        plt.ylabel("Keyword")
        plt.tight_layout()
        plt.savefig("output/top_keywords.png")
        plt.close()

        top_words_df.to_csv("data/top_keywords.csv", index=False)
        filtered_df[["title", "sentiment", "sentiment_confidence"]].to_csv("data/headlines_with_sentiment.csv", index=False)

    print("Sentiment and keyword analysis done. Charts and CSVs saved.")

if __name__ == "__main__":
    main()
