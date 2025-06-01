import os

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
import joblib
import nltk
from nltk.corpus import stopwords, words

from preprocess import Preprocess

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('words')
english_words = set(words.words())

class LDA:
    def __init__(self, df):
        self.df = df

    def train(self):
        self.vectorizer = TfidfVectorizer(
            max_df=0.8,
            min_df=5,
            stop_words='english',
            ngram_range=(1, 2)
        )

        text = self.df['clean_text'].to_list()

        text_transform = self.vectorizer.fit_transform(text)

        n_topics = 10
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method='batch',
            random_state=0
        )
        self.lda.fit(text_transform)

    def printTopics(self, model, vectorizer):
        top_topic = 10
        words = vectorizer.get_feature_names_out()
        for i, topic in enumerate(model.components_):
            top_features = topic.argsort()[::-1][:10]
            topic_words = [words[i] for i in top_features]
            print(f"Topic #{i + 1:02d}: ", ', '.join(topic_words))

class BERTOPIC:
    def __init__(self, df):
        self.df = df
        self.embedding_model_path = "embedding_model"
        self.embeddings_path = "embeddings.pt"
        self.embeddings_path_cpu = "embeddings_cpu.pt"

    def elbow(self):
        sse = []
        k_range = range(2, 51)

        for k in k_range:
            print(f"\rElbow progress: k={k}", end="")
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(self.embeddings_cpu)
            sse.append(kmeans.inertia_)

        plt.figure(figsize=(8, 6))
        plt.plot(k_range, sse, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters')
        plt.ylabel('SSE')
        plt.show()

    def embed(self):
        # self.vectorizer_model = CountVectorizer(stop_words='english', ngram_range=(1, 2))
        print("Training embedding model")
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')

        def only_english_words(text):
            words = text.split()
            filtered_words = [word for word in words if word.lower() in english_words]
            return ' '.join(filtered_words)

        text = self.df['clean_text'].tolist()
        self.text_with_stopwords = [
            only_english_words(' '.join([word for word in doc.split() if word.lower() not in stop_words]))
            for doc in text
        ]

        if os.path.exists(self.embeddings_path_cpu):
            print("Loading embeddings")
            self.embeddings_cpu = torch.load(self.embeddings_path_cpu)
        else:
            print("Encoding embeddings")
            self.embeddings_cpu = self.embedding_model.encode(self.text_with_stopwords, convert_to_tensor=False)
            torch.save(self.embeddings_cpu, self.embeddings_path_cpu)

    def train_kmeans_bertopic(self):
        # kmeans
        print("Clustering")
        best_k = 15
        kmeans = KMeans(n_clusters=best_k, random_state=0)
        clusters = kmeans.fit_predict(self.embeddings_cpu)
        self.df['cluster'] = clusters

        # bertopic
        # hdbscan_model = HDBSCAN(memory=None)
        topic_model = BERTopic(embedding_model=None,
                               # hdbscan_model=hdbscan_model,
                               )
        for cluster_id in range(best_k):
            cluster_data = self.df[self.df['cluster'] == cluster_id]['clean_text'].tolist()
            if len(cluster_data) > 10:
                topics, probs = topic_model.fit_transform(documents=cluster_data, embeddings=self.embeddings_cpu)
                print(f"Cluster #{cluster_id} Topics:")
                print(topic_model.get_topic_info())

    def train_bertopic(self):
        # bertopic
        # hdbscan_model = HDBSCAN(memory=None)
        # memory = joblib.Memory(location=None)
        custom_vector = CountVectorizer(stop_words="english",
                                     ngram_range=(1, 2),
                                     max_df=0.8,
                                     min_df=5)
        topic_model = BERTopic(embedding_model=None,
                               # hdbscan_model=hdbscan_model,
                               vectorizer_model=custom_vector)
        topics, probs = topic_model.fit_transform(documents=self.text_with_stopwords, embeddings=self.embeddings_cpu)
        print(topic_model.get_topic_info().head(15))
        joblib.dump(topic_model, "bertopic_model.pkl.lzma", compress=("lzma", 9))

    def load_bertopic(self):
        topic_model = joblib.load("bertopic_model.pkl.lzma")
        embeddings_cpu = torch.load(self.embeddings_path_cpu)
        def only_english_words(text):
            words = text.split()
            filtered_words = [word for word in words if word.lower() in english_words]
            return ' '.join(filtered_words)

        text = self.df['clean_text'].tolist()
        text_with_stopwords = [
            only_english_words(' '.join([word for word in doc.split() if word.lower() not in stop_words]))
            for doc in text
        ]
        topics, probabilities = topic_model.transform(text_with_stopwords, embeddings=embeddings_cpu)
        print(topic_model.get_topic_info().head(15))

        


if __name__ == '__main__':
    path = './money_df_clean_text.parquet'
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        preprocess = Preprocess()
        preprocess.loadData()
        preprocess.cleanData()
        df = preprocess.money_df


    # lda = LDA(df)
    # lda.train()
    # lda.printTopics(lda.lda, lda.vectorizer)

    bertopic = BERTOPIC(df)
    if os.path.exists("bertopic_model.pkl.lzma"):
        bertopic.load_bertopic()
    else:
        bertopic.embed()
        # bertopic.elbow()
        bertopic.train_bertopic()
