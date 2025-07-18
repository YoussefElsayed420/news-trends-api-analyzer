import requests
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    API_KEY = "6b1237181ac54545ba30de67846a9ef9"
    URL = "https://newsapi.org/v2/top-headlines"
    PARAMS = {
        "country": "us",
        "pageSize": 100,
        "apiKey": API_KEY
    }
    # Request the latest news headlines
    response = requests.get(URL, params=PARAMS)
    data = response.json()
    articles = data["articles"]
    # Convert the articles to a DataFrame for analysis
    df = pd.DataFrame(articles)
    # Extract the source name from the nested dictionary
    df["source"] = df["source"].apply(lambda x: x["name"])
    # Parse the publication date
    df["publishedAt"] = pd.to_datetime(df["publishedAt"])
    # Keep only the columns we care about
    df = df.loc[:, ["source", "author", "title", "description", "publishedAt", "url"]].copy()
    # Add an 'hour' column for time-based analysis
    df["hour"] = df["publishedAt"].dt.hour

    # Make sure output folders exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Set seaborn style for all plots
    sns.set(style="whitegrid")

    # Plot the top 10 news sources by article count
    df["source"].value_counts().head(10).plot(kind="barh", title="Top News Sources")
    plt.tight_layout()
    plt.savefig("output/top_sources.png")
    plt.clf()

    # Plot the distribution of article publication times by hour
    df["hour"].value_counts().sort_index().plot(kind="line", title="Publishing Frequency by Hour")
    plt.xlabel("Hour")
    plt.tight_layout()
    plt.savefig("output/publish_hours.png")
    plt.clf()

    # Generate a word cloud from all the news headlines
    text = " ".join(pd.Series(df["title"]).dropna().tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Headline WordCloud")
    plt.tight_layout()
    plt.savefig("output/headline_wordcloud.png")
    plt.clf()

    # Save the cleaned and processed headlines to CSV
    df.to_csv("data/top_headlines.csv", index=False)

    print("✅ News fetch and analysis complete. Check output/ and data/ folders.")

if __name__ == "__main__":
    main() 