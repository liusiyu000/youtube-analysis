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
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from Q1.preprocess import Preprocess

# 下载 NLTK 资源
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
        self.embeddings_path = "embeddings.pt"
        self.embeddings_path_cpu = "embeddings_cpu.pt"

    def embed(self):
        """
        This method will generate and save text embeddings using SentenceTransformer
        """
        print("Training embedding model")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')  # GPU

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
            self.embeddings = self.embedding_model.encode(self.text_with_stopwords, convert_to_tensor=True, batch_size=256, show_progress_bar=True)  # GPU
            self.embeddings_cpu = self.embeddings.cpu().numpy()
            torch.save(self.embeddings_cpu, self.embeddings_path_cpu)

    def train_bertopic(self):
        """
        This method will train a BERTopic model on the provided data and save the model
        """
        # 使用 CountVectorizer 进行特征提取
        print("Training topic model")
        custom_vector = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.8, min_df=5)

        # 训练 BERTopic 模型
        topic_model = BERTopic(embedding_model=None, vectorizer_model=custom_vector)
        topics, probs = topic_model.fit_transform(documents=self.text_with_stopwords, embeddings=self.embeddings_cpu)

        # 输出主题信息
        print(topic_model.get_topic_info().head(15))
        joblib.dump(topic_model, "bertopic_model_old.pkl.lzma", compress=("lzma", 3))
        return topic_model, topics, probs


    def load_bertopic(self):
        """
        This method will load the trained BERTopic model and apply it to the data.
        """
        topic_model = joblib.load("bertopic_model_old.pkl.lzma")
        embeddings_cpu = torch.load(self.embeddings_path_cpu)

        # 重新获取文本
        text = self.df['clean_text'].tolist()
        text_with_stopwords = [
            only_english_words(' '.join([word for word in doc.split() if word.lower() not in stop_words]))
            for doc in text
        ]

        # 主题转换
        topics, probabilities = topic_model.transform(text_with_stopwords, embeddings=embeddings_cpu)
        print(topic_model.get_topic_info().head(15))
        return topic_model, topics, probabilities


def get_topic_labels(topic_model):
    """
    Extracts the topic labels automatically based on the top words in each topic
    """
    topic_info = topic_model.get_topic_info()
    topic_labels = {}

    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id != -1:  # 排除噪声类
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                # 基于关键词创建标签，可以根据业务需求调整
                topic_label = " ".join([word for word, _ in topic_words[:5]])  # 取前5个关键词
                topic_labels[topic_id] = topic_label
    return topic_labels


def filter_money_related_topics(topic_model, topics, keywords=MONEY_KEYWORDS):
    """
    Filters topics that are related to 'money', 'business', etc.
    """
    money_related_topics = {}

    for topic_id, topic in zip(topics, topics):
        topic_words = topic_model.get_topic(topic_id)
        topic_keywords = [word for word, _ in topic_words]
        if any(keyword in topic_keywords for keyword in keywords):
            money_related_topics[topic_id] = topic_keywords
    return money_related_topics


def compute_cv_coh(df, topic_model):
    text = df['clean_text'].tolist()
    text_with_stopwords = [
        only_english_words(' '.join([word for word in doc.split() if word.lower() not in stop_words]))
        for doc in text
    ]
    tokens = [doc.split() for doc in text_with_stopwords]

    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(tok) for tok in tokens]

    topic_words = []
    n_top = 20
    for topic_id in topic_model.get_topic_info()["Topic"].unique():
        if topic_id == -1:
            continue
        words_scores = topic_model.get_topic(topic_id)
        topic = []
        for word, _ in words_scores:  # top-20
            word = word.strip()
            if word in dictionary.token2id:
                topic.append(word)
        if topic:
            topic_words.append(topic)

    cm = CoherenceModel(
        topics=topic_words,
        texts=tokens,
        dictionary=dictionary,
        corpus=corpus,
        coherence='c_v',
    )

    score = cm.get_coherence()
    print(f"Top 20 topic Cv Coherence: {score:.4f}")


def main():
    path = './money_df_clean_text.parquet'  # 数据路径
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        preprocess = Preprocess()  # 假设你有一个清洗数据的类
        preprocess.loadData()
        preprocess.cleanData()
        df = preprocess.df

    # 初始化 BERTOPIC 类
    bertopic = BERTOPIC(df)

    # 加载预训练的 BERTopic 模型
    if os.path.exists("bertopic_model_old.pkl.lzma"):
        topic_model, topics, _ = bertopic.load_bertopic()
    else:
        # 如果模型不存在，则进行嵌入和训练
        bertopic.embed()  # 生成嵌入
        topic_model, topics, _ = bertopic.train_bertopic()  # 训练 BERTopic


    compute_cv_coh(df, topic_model)





visualization = False

if __name__ == '__main__':
    if visualization:
        pass
    else:
        main()


