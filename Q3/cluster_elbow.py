import re
from urllib.parse import urlparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nltk.corpus import stopwords, words
from sklearn.cluster import KMeans, MiniBatchKMeans
from kneed import KneeLocator

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

stop_words = set(stopwords.words('english'))
english_words = set(words.words())

SAMPLE = False


class RevenueAnalyzer:
    def __init__(self):
        self.df_videos = None
        self.df_channels = None
        self.video_path = "../Q1/money_related_content.jsonl"

        self.channel_path = r"F:\\dissertationData\\df_channels_en.tsv"
        self.topic_model_path = "../Q1/bertopic_model.pkl.lzma"
        self.embeddings_path = "../Q1/embeddings_cpu.pt"
        self.money_df_path = "../Q1/money_df_clean_text.parquet"

        self.revenue_patterns = {
            "affiliate": [
                "affiliate",
                "referral",
                "commission",
                "promo code",
                "discount code",
                "coupon",
                "link in bio",
                "link below",
                "use my link",
                "special offer",
            ],
            "sponsored": [
                "sponsored",
                "partnership",
                "collaboration",
                "brand ambassador",
                "paid promotion",
                "thanks to",
                "supported by",
                "brought to you by",
            ],
            "course": [
                "course",
                "training",
                "masterclass",
                "bootcamp",
                "workshop",
                "coaching",
                "consultation",
                "mentoring",
                "ebook",
            ],
            "product": [
                "merch",
                "merchandise",
                "shop",
                "store",
                "app",
                "software",
                "subscription",
                "membership",
                "premium",
            ],
            "donation": ["patreon", "paypal", "donate", "support", "tip jar", "ko-fi"],
        }

        self.affiliate_domains = ["amazon.com", "amzn.to", "bit.ly", "geni.us", "clickbank.net"]

        self.df_videos = None
        self.df_channels = None
        self.revenue_result = {}
        self.platform_counts = {}
        self.channel_stats = {}
        self.external_videos_with_topics = None
        self.revenue_clusters = None
        self.strategy_labels = {}

    def _generate_strategy_labels(self, max_len=14):
        """
        cluster_id → readable label
        - 取该簇出现率最高的 2 种 revenue_type 组合成标签
        - 超过 max_len 就自动换行，便于在图里显示
        """
        self.strategy_labels = {}
        for cid, data in self.revenue_clusters.items():
            profile = data.get("profile", {})
            top_types = [rt for rt, v in sorted(profile.items(),
                                                key=lambda x: x[1],
                                                reverse=True) if v > 0][:2]
            label = "_".join(top_types) if top_types else f"cluster_{cid}"
            if len(label) > max_len:
                parts = label.split("_")
                label = "\n".join(["_".join(parts[:1]), "_".join(parts[1:])])
            self.strategy_labels[cid] = label



    def load_data(self):
        print("Loading data...")
        self.df_videos = pd.read_json(self.video_path, lines=True)
        self.df_channels = pd.read_csv(self.channel_path, sep="\t")

        self.df_videos["description"] = self.df_videos["description"].fillna("")
        self.df_videos["title"] = self.df_videos["title"].fillna("")

    def analyze_revenue_methods(self):
        def detect_keywords(text_column, keywords):
            pattern = "|".join(keywords)
            return self.df_videos[text_column].str.contains(pattern, case=False, na=False)

        def extract_links(description):
            if not description:
                return []
            url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            return re.findall(url_pattern, description)

        print("Analyzing revenue methods...")

        for revenue_type, keywords in self.revenue_patterns.items():
            col_name = f"has_{revenue_type}"
            self.df_videos[col_name] = detect_keywords("description", keywords) | detect_keywords("title", keywords)
            self.revenue_result[revenue_type] = {
                "count": self.df_videos[col_name].sum(),
                "percentage": self.df_videos[col_name].mean(),
            }

        self.df_videos["links"] = self.df_videos["description"].apply(extract_links)
        self.df_videos["num_links"] = self.df_videos["links"].apply(len)

    def analyze_external_platforms(self):
        print("Analyzing external platforms...")

        def extract_domain(url):
            try:
                return urlparse(url).netloc.replace("www.", "")
            except Exception:
                return ""

        all_links = self.df_videos["links"].explode().dropna()
        domains = all_links.apply(extract_domain)
        self.platform_counts = domains.value_counts().to_dict()

    def channel_analysis(self):
        print("Channel level analysis...")

        revenue_cols = [f"has_{rt}" for rt in self.revenue_patterns.keys()]
        channel_group = self.df_videos.groupby("channel_id")

        revenue_diversity = channel_group[revenue_cols].sum().gt(0).sum(axis=1)
        view_count_sum = channel_group["view_count"].sum()

        self.channel_stats = pd.DataFrame(
            {
                "channel_id": revenue_diversity.index,
                "revenue_diversity_score": revenue_diversity.values,
                "view_count_sum": view_count_sum.values,
            }
        )

    def save_results(self):
        merged_channel_data = pd.merge(
            self.channel_stats,
            self.df_channels[["channel", "name_cc"]],
            left_on="channel_id",
            right_on="channel",
            how="inner",
        ).drop(columns="channel_id")
        merged_channel_data.to_csv("channel_revenue_analysis.csv", index=False)

        key_features = ["display_id", "channel_id", "title", "view_count"] + [
            f"has_{rt}" for rt in self.revenue_patterns.keys()
        ] + ["num_links"]
        self.df_videos[key_features].to_csv("video_revenue_features.csv", index=False)


    def topic_model_strategy(self):
        print("Topic based external strategy analysis")

        topic_model = joblib.load(self.topic_model_path)
        embeddings_cpu = torch.load(self.embeddings_path)
        df = pd.read_parquet(self.money_df_path)

        def only_english_words(text):
            words_ = text.split()
            filtered_words = [word for word in words_ if word.lower() in english_words]
            return " ".join(filtered_words)

        text = df["clean_text"].tolist()
        text_with_stopwords = [
            only_english_words(" ".join([w for w in doc.split() if w.lower() not in stop_words])) for doc in text
        ]

        topics, probabilities = topic_model.transform(text_with_stopwords, embeddings=embeddings_cpu)
        self.df_videos["topic"] = topics
        external_videos = self.df_videos[self.df_videos["num_links"] > 0].copy()

        topic_info = topic_model.get_topic_info()
        n_topics = len(topic_info) - 1

        self.topic_revenue_analysis = {}
        for topic_id in topic_info["Topic"].unique():
            if topic_id == -1:
                continue
            topic_videos = external_videos[external_videos["topic"] == topic_id]
            revenue_method_counts = {}
            for rt in self.revenue_patterns.keys():
                col_name = f"has_{rt}"
                revenue_method_counts[rt] = topic_videos[col_name].sum()
            top_terms = [w for w, _ in topic_model.get_topic(topic_id)[:2]]
            topic_readable = " ".join(top_terms)
            self.topic_revenue_analysis[topic_id] = {
                "topic_name": topic_readable,
                "video_count": len(topic_videos),
                "revenue_methods": revenue_method_counts,
            }

        revenue_matrix = external_videos[[f"has_{rt}" for rt in self.revenue_patterns.keys()]].values

        k_range = range(2, 50)
        inertias = []

        for k in k_range:
            mb_kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=10000)
            mb_kmeans.fit(revenue_matrix)
            inertias.append(mb_kmeans.inertia_)

        plt.figure(figsize=(12, 8))
        plt.plot(k_range, inertias, marker='o')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.title("Elbow Method for Optimal k")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("kmeans_elbow_plot.png", dpi=600)
        plt.show()

        kl = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
        optimal_k = kl.elbow
        print(f"KneeLocator-selected optimal k: {optimal_k}")

if __name__ == "__main__":
    analyzer = RevenueAnalyzer()
    analyzer.load_data()
    analyzer.analyze_revenue_methods()
    analyzer.analyze_external_platforms()
    analyzer.channel_analysis()
    analyzer.save_results()

    analyzer.topic_model_strategy()
