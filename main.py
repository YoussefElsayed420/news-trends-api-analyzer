from src.fetch_news import main as fetch_news_main
from src.sentiment_analysis import main as sentiment_analysis_main
from src.topic_clustering import main as topic_clustering_main
from src.label_clusters import main as label_clusters_main
from src.visualize_named_clusters import main as visualize_named_clusters_main

if __name__ == "__main__":
    fetch_news_main()
    sentiment_analysis_main()
    topic_clustering_main()
    label_clusters_main()
    visualize_named_clusters_main() 