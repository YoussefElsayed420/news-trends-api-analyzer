import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Load the labeled clusters
    df = pd.read_csv("data/clustered_headlines_labeled.csv")

    # Count headlines per named cluster
    label_counts = df['label'].value_counts().reset_index()
    label_counts.columns = ['label', 'headline_count']

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=label_counts, x='label', y='headline_count', palette='tab10')
    plt.title('Number of Headlines per Named Cluster')
    plt.xlabel('Cluster Label (Top Keywords)')
    plt.ylabel('Number of Headlines')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/named_topic_summary.png')
    plt.close()

    print("✅ Named cluster summary plot saved to output/named_topic_summary.png")

if __name__ == "__main__":
    main() 