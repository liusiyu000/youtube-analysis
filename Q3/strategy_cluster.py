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

    def visualization(self):
        def remove_outliers(data):
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data >= lower_bound) & (data <= upper_bound)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        revenue_data = {k.capitalize(): v["count"] for k, v in self.revenue_result.items()}
        axes[0, 0].pie(revenue_data.values(), labels=revenue_data.keys(), autopct="%1.1f%%")
        axes[0, 0].set_title("Revenue method")

        platforms = list(self.platform_counts.keys())[:8]
        counts = [self.platform_counts[p] for p in platforms]
        axes[0, 1].bar(platforms, counts)
        axes[0, 1].set_title("External links")
        axes[0, 1].tick_params(axis="x", rotation=45)

        diversity_dist = self.channel_stats["revenue_diversity_score"].value_counts().sort_index()
        axes[1, 0].bar(diversity_dist.index, diversity_dist.values)
        axes[1, 0].set_title("Channel Revenue Diversity")
        axes[1, 0].set_xlabel("Revenue Count")

        outlier_mask = remove_outliers(self.channel_stats["view_count_sum"])
        filtered_data = self.channel_stats[outlier_mask]
        filtered_data["view_count_sum"].replace(0, 1)

        sns.boxenplot(x="revenue_diversity_score", y="view_count_sum", data=filtered_data)
        plt.yscale("log")
        axes[1, 1].set_xlabel("Channel Revenue Diversity")
        axes[1, 1].set_ylabel("View Count")
        axes[1, 1].set_title("Diversity VS View count")

        plt.tight_layout()
        plt.savefig("revenue_analysis.png", dpi=600)
        plt.show()

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
        n_clusters = 18
        print(len(external_videos)) # 470591
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=10000, n_init="auto")
        external_videos["revenue_cluster"] = kmeans.fit_predict(revenue_matrix)

        self.revenue_clusters = {}
        for cluster_id in range(n_clusters):
            cluster_videos = external_videos[external_videos["revenue_cluster"] == cluster_id]

            cluster_profile = {}
            for rt in self.revenue_patterns.keys():
                col_name = f"has_{rt}"
                if col_name in cluster_videos.columns:
                    cluster_profile[rt] = cluster_videos[col_name].mean()

            top_topics = cluster_videos["topic"].value_counts().head(3)

            self.revenue_clusters[cluster_id] = {
                "size": len(cluster_videos),
                "profile": cluster_profile,
                "avg_links": cluster_videos["num_links"].mean(),
                "avg_views": cluster_videos["view_count"].mean(),
                "top_topics": top_topics.to_dict(),
            }

        self._generate_strategy_labels()

        self.external_videos_with_topics = external_videos

    def visualization_topic(self):
        if not (hasattr(self, "topic_revenue_analysis") and self.topic_revenue_analysis):
            return

        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

        revenue_types = list(self.revenue_patterns.keys())
        cluster_labels = [self.strategy_labels.get(i, f"cluster_{i}")
                          for i in range(len(self.revenue_clusters))]
        matrix = np.array([[self.revenue_clusters[i]["profile"].get(rt, 0)
                            for i in range(len(self.revenue_clusters))]
                           for rt in revenue_types])

        sns.heatmap(pd.DataFrame(matrix,
                                 index=revenue_types,
                                 columns=cluster_labels),
                    ax=axes2[0, 0],
                    cmap="YlGnBu",
                    vmin=0, vmax=1,
                    annot=True, fmt=".2f")
        axes2[0, 0].set_title("Revenue-type distribution per Strategy cluster")
        axes2[0, 0].tick_params(axis="x", rotation=45)

        top_topics = sorted(self.topic_revenue_analysis.items(),
                            key=lambda x: x[1]["video_count"],
                            reverse=True)[:10]

        topic_labels = [data["topic_name"] for _, data in top_topics]
        topic_matrix = np.array([[data["revenue_methods"].get(rt, 0)
                                  for rt in revenue_types] for _, data in top_topics]).T

        sns.heatmap(pd.DataFrame(topic_matrix,
                                 index=revenue_types,
                                 columns=topic_labels),
                    ax=axes2[0, 1],
                    cmap="YlOrRd",
                    annot=True, fmt="d")
        axes2[0, 1].set_title("Topics and ways to make money")
        axes2[0, 1].tick_params(axis="x", rotation=45)

        sizes = [v["size"] for v in self.revenue_clusters.values()]
        axes2[1, 0].bar(cluster_labels, sizes)
        axes2[1, 0].set_title("Video count per Strategy cluster")
        axes2[1, 0].set_ylabel("video numbers")
        axes2[1, 0].tick_params(axis="x", rotation=45)


        views = [v["avg_views"] for v in self.revenue_clusters.values()]
        axes2[1, 1].bar(cluster_labels, views)
        axes2[1, 1].set_title("Average views per Strategy cluster")
        axes2[1, 1].set_ylabel("Average view count")
        axes2[1, 1].tick_params(axis="x", rotation=45)


        plt.savefig("topic_revenue_analysis.png", dpi=600)
        plt.show()


if __name__ == "__main__":
    analyzer = RevenueAnalyzer()
    analyzer.load_data()
    analyzer.analyze_revenue_methods()
    analyzer.analyze_external_platforms()
    analyzer.channel_analysis()
    analyzer.visualization()
    analyzer.save_results()

    analyzer.topic_model_strategy()
    analyzer.visualization_topic()
