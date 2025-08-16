import os

import joblib
import nltk
import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from nltk.corpus import stopwords, words
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim

from BertopicVisualizer import BERTopicVisualizer
from Q1.preprocess import Preprocess

nltk.download('stopwords')
nltk.download('words')
stop_words = set(stopwords.words('english'))
english_words = set(words.words())
MONEY_KEYWORDS = [
    "money","income","earning","profit","revenue",
    "dropshipping","ecommerce","marketing alliance", "freelance","investing",
    "cryptocurrency","stock","trading", "$", "dollar", "hustle", "wealth",
    "shopify","sponsorship", "investment",
]

def only_english_words(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() in english_words]
    return ' '.join(filtered_words)

class BERTOPIC:
    def __init__(self, df):
        self.df = df
        self.embeddings.pt = "embeddings.pt"
        self.embeddings_path_cpu = "embeddings_cpu.pt"

    def embed(self):
        print("Training embedding model")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

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
            self.embeddings = self.embedding_model.encode(self.text_with_stopwords, convert_to_tensor=True, batch_size=256, show_progress_bar=True)
            self.embeddings_cpu = self.embeddings.cpu().numpy()
            torch.save(self.embeddings_cpu, self.embeddings_path_cpu)

    def train_bertopic(self):

        custom_vector = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.8, min_df=5)

        topic_model = BERTopic(embedding_model=None, vectorizer_model=custom_vector)
        topics, probs = topic_model.fit_transform(documents=self.text_with_stopwords, embeddings=self.embeddings_cpu)

        print(topic_model.get_topic_info().head(15))
        joblib.dump(topic_model, "bertopic_model.pkl.lzma", compress=("lzma", 3))
        return topic_model, topics, probs


    def load_bertopic(self):

        topic_model = joblib.load("bertopic_model.pkl.lzma")
        embeddings_cpu = torch.load(self.embeddings_path_cpu)

        text = self.df['clean_text'].tolist()
        text_with_stopwords = [
            only_english_words(' '.join([word for word in doc.split() if word.lower() not in stop_words]))
            for doc in text
        ]

        topics, probabilities = topic_model.transform(text_with_stopwords, embeddings=embeddings_cpu)
        print(topic_model.get_topic_info().head(15))
        return topic_model, topics, probabilities


def get_topic_labels(topic_model):

    topic_info = topic_model.get_topic_info()
    topic_labels = {}

    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id != -1:
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:

                topic_label = " ".join([word for word, _ in topic_words[:5]])
                topic_labels[topic_id] = topic_label
    return topic_labels


def filter_money_related_topics(topic_model, topics, keywords=MONEY_KEYWORDS):

    money_related_topics = {}

    for topic_id, topic in zip(topics, topics):
        topic_words = topic_model.get_topic(topic_id)
        topic_keywords = [word for word, _ in topic_words]
        if any(keyword in topic_keywords for keyword in keywords):
            money_related_topics[topic_id] = topic_keywords
    return money_related_topics


def save_topic_details(topic_model, topic_labels, save_path="topic_details.xlsx"):

    with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
        topic_info = topic_model.get_topic_info()
        topic_info.to_excel(writer, sheet_name='Overview', index=False)

        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:
                topic_words = topic_model.get_topic(topic_id)[:20]
                df_words = pd.DataFrame(topic_words, columns=['Word', 'Score'])
                df_words['Rank'] = range(1, len(df_words) + 1)
                df_words = df_words[['Rank', 'Word', 'Score']]
                df_words.to_excel(writer, sheet_name=f'Topic_{topic_id}', index=False)

def main():
    path = './money_df_clean_text.parquet'
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        preprocess = Preprocess()
        preprocess.loadData()
        preprocess.cleanData()
        df = preprocess.df

    bertopic = BERTOPIC(df)

    if os.path.exists("bertopic_model.pkl.lzma"):
        topic_model, topics, _ = bertopic.load_bertopic()
    else:
        bertopic.embed()
        topic_model, topics, _ = bertopic.train_bertopic()

    topic_labels = get_topic_labels(topic_model)
    save_topic_details(topic_model, topic_labels, save_path="topic_details.xlsx")

    money_related_topics = filter_money_related_topics(topic_model, topics)



visualization = False

if __name__ == '__main__':
    if visualization:
        viz = BERTopicVisualizer(model_path="bertopic_model.pkl.lzma")
        viz.create_wordcloud_grid(n_topics=15, save_path="topic_wordclouds.png")
        viz.create_topic_heatmap(n_topics=10, n_words=5, save_path="topic_heatmap.png")
        viz.create_interactive_topic_viz(n_topics=15, save_path="interactive_topics.html")
    else:
        main()


