import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys
from scipy.sparse import issparse

def main():
    input_path = "data/clustered_headlines.csv"
    output_path = "data/clustered_headlines_labeled.csv"

    if not os.path.exists(input_path):
        print(f"error: {input_path} not found. Please ensure the clustering step runs successfully before labeling clusters.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    if df.empty or "title" not in df.columns or "cluster" not in df.columns:
        print("error: Input file is empty or missing required columns.")
        sys.exit(1)

    cluster_texts = df.groupby("cluster")["title"].apply(lambda x: " ".join(x)).reset_index()

    if cluster_texts.empty:
        print("error: No clusters found in input file.")
        sys.exit(1)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(cluster_texts["title"])
    terms = vectorizer.get_feature_names_out()

    def get_top_keywords(row_idx: int, top_n: int = 3) -> list:
        row = tfidf_matrix[int(row_idx)]
        # Only call .toarray() if row is a sparse matrix
        if issparse(row):
            from scipy.sparse import csr_matrix
            row = row if isinstance(row, csr_matrix) else csr_matrix(row)
            row_vec = row.toarray().flatten()
        else:
            return []
        top_indices = row_vec.argsort()[::-1][:top_n]
        return [str(terms[i]) for i in top_indices if row_vec[i] > 0]

    # Use integer indices to access tfidf_matrix rows
    cluster_texts["label"] = [
        " & ".join(get_top_keywords(idx)) if get_top_keywords(idx) else "Unlabeled"
        for idx in range(len(cluster_texts))
    ]

    df = df.merge(cluster_texts[["cluster", "label"]], on="cluster", how="left")
    df.to_csv(output_path, index=False)
    print(f"Labels generated and saved to {output_path}")

    print("\nCluster Summary:")
    for cluster_num, group in df.groupby("cluster"):
        label = group["label"].iloc[0]
        print(f"\nCluster {cluster_num} ({label}) - {len(group)} headlines:")
        for i, title in enumerate(group["title"].head(3)):
            print(f"  Example {i+1}: {title}")

if __name__ == "__main__":
    main() 