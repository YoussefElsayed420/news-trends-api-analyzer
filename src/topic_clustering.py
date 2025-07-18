import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import nltk
import os
import sys
import string
nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def main():
    input_path = 'data/top_headlines.csv'
    if not os.path.exists(input_path):
        print(f"❌ ERROR: {input_path} not found. Please ensure the news fetch step runs successfully before clustering.")
        sys.exit(1)
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['title']).copy()
    # Preprocess headlines
    df['clean_title'] = df['title'].apply(preprocess_text)

    # Vectorize with TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['clean_title'])

    # Cluster headlines (choose 7 clusters as a middle ground)
    n_clusters = 7
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    # Save clustered headlines
    df[['title', 'cluster']].to_csv('data/clustered_headlines.csv', index=False)

    # Plot number of headlines per cluster
    os.makedirs('output/cluster_wordclouds', exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.countplot(x='cluster', data=df, palette='tab10')
    plt.title('Number of Headlines per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Headline Count')
    plt.tight_layout()
    plt.savefig('output/topic_summary.png')
    plt.close()

    # Generate a wordcloud for each cluster
    for cluster_num in range(n_clusters):
        cluster_text = ' '.join(df[df['cluster'] == cluster_num]['clean_title'])
        if cluster_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Cluster {cluster_num} WordCloud')
            plt.tight_layout()
            plt.savefig(f'output/cluster_wordclouds/cluster_{cluster_num}.png')
            plt.close()

    print('✅ Topic clustering complete. Results saved to data/ and output/.')

if __name__ == '__main__':
    main() 