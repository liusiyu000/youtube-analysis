import os

import joblib
import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from urllib.parse import urlparse

import torch
from nltk.corpus import stopwords, words
from sklearn.cluster import KMeans

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
        self.video_path = 'F:\dissertationData\yt_metadata_en_50k.jsonl' if SAMPLE else 'F:\dissertationData\yt_metadata_en_money.jsonl'
        self.channel_path = r'F:\dissertationData\top_1000_df_channels_en.tsv' if SAMPLE else 'F:\dissertationData\df_channels_en.tsv'
        self.topic_model_path = "../Q1/bertopic_model.pkl.lzma"
        self.embeddings_path = "../Q1/embeddings_cpu.pt"
        self.money_df_path = '../Q1/money_df_clean_text.parquet'

        self.revenue_patterns = {
            'affiliate': [
                'affiliate', 'referral', 'commission', 'promo code', 'discount code',
                'coupon', 'link in bio', 'link below', 'use my link', 'special offer'
            ],
            'sponsored': [
                'sponsored', 'partnership', 'collaboration', 'brand ambassador',
                'paid promotion', 'thanks to', 'supported by', 'brought to you by'
            ],
            'course': [
                'course', 'training', 'masterclass', 'bootcamp', 'workshop',
                'coaching', 'consultation', 'mentoring', 'ebook'
            ],
            'product': [
                'merch', 'merchandise', 'shop', 'store', 'app', 'software',
                'subscription', 'membership', 'premium'
            ],
            'donation': [
                'patreon', 'paypal', 'donate', 'support', 'tip jar', 'ko-fi'
            ]
        }

        self.affiliate_domains = [
            'amazon.com', 'amzn.to', 'bit.ly', 'geni.us', 'clickbank.net'
        ]

        self.external_platforms = {
            'Instagram': ['instagram.com', 'ig.me'],
            'Twitter': ['twitter.com', 't.co'],
            'Patreon': ['patreon.com'],
            'Discord': ['discord.gg'],
            'Website': ['www.', '.com']
        }

    def load_data(self):
        self.df_videos = pd.read_json(self.video_path, lines=True)
        self.df_channels = pd.read_csv(self.channel_path, sep='\t')

        self.df_videos['description'] = self.df_videos['description'].fillna('')
        self.df_videos['title'] = self.df_videos['title'].fillna('')

    def analyze_revenue_methods(self):
        def detect_keywords(text_column, keywords):
            # general keyword
            pattern = '|'.join(keywords)
            return self.df_videos[text_column].str.contains(pattern, case=False, na=False)

        def extract_links(description):
            if not description:
                return []
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            return re.findall(url_pattern, description)

        def has_affiliate_domain(links):
            if not links:
                return False
                for link in links:
                    try:
                        domain = urlparse(link).netloc.lower()
                    except:
                        continue
                    if any(affiliate_domain in domain for affiliate_domain in self.affiliate_domains):
                        return True
            return False

        self.revenue_result = {}

        # affiliate marketing
        self.df_videos['links'] = self.df_videos['description'].apply(extract_links)
        self.df_videos['num_links'] = self.df_videos['links'].apply(len)
        self.df_videos['has_affiliate_links'] = self.df_videos['links'].apply(has_affiliate_domain)

        # revenue detect
        for revenue_type, keywords in self.revenue_patterns.items():
            self.df_videos[f'has_{revenue_type}'] = detect_keywords('title', keywords) | detect_keywords('description', keywords)

            if revenue_type == 'affiliate':
                self.df_videos[f'has_{revenue_type}'] = (
                        self.df_videos[f'has_{revenue_type}'] | self.df_videos['has_affiliate_links']
                )

            count = self.df_videos[f'has_{revenue_type}'].sum()
            percentage = count / len(self.df_videos) * 100
            self.revenue_result[revenue_type] = {'count': count, 'percentage': percentage}

            print(f"{revenue_type.capitalize()}: {count:,} ({percentage:.2f}%)")

    def analyze_external_platforms(self):
        self.platform_counts = {}
        for platform, domains in self.external_platforms.items():
            count = sum(self.df_videos['description'].str.contains(domain, case=False, na=False).sum()
                        for domain in domains)
            self.platform_counts[platform] = count

    def channel_analysis(self):
        agg_dict = {
            'view_count': ['sum', 'mean'],
            'like_count': ['sum', 'mean'],
            'num_links': 'mean',
            'display_id': 'count'
        }

        for revenue_type in self.revenue_patterns.keys():
            agg_dict[f'has_{revenue_type}'] = 'sum'

        self.channel_stats = self.df_videos.groupby('channel_id').agg(agg_dict).round(2)
        self.channel_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0]
                                 for col in self.channel_stats.columns]
        self.channel_stats = self.channel_stats.reset_index()

        revenue_cols = [f'has_{revenue_type}_sum' for revenue_type in self.revenue_patterns.keys()]
        self.channel_stats['revenue_diversity_score'] = (self.channel_stats[revenue_cols] > 0).sum(axis=1)

        for col in revenue_cols:
            rate_col = col.replace('_sum', '_rate')
            self.channel_stats[rate_col] = (self.channel_stats[col] / self.channel_stats['display_id_count']).round(3)

    def visualization(self):
        def remove_outliers(data):
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data >= lower_bound) & (data <= upper_bound)

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        revenue_data = {k.capitalize(): v['count'] for k, v in self.revenue_result.items()}
        axes[0, 0].pie(revenue_data.values(), labels=revenue_data.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Revenue method')

        platforms = list(self.platform_counts.keys())[:8]
        counts = [self.platform_counts[p] for p in platforms]
        axes[0, 1].bar(platforms, counts)
        axes[0, 1].set_title('External links')
        axes[0, 1].tick_params(axis='x', rotation=45)

        diversity_dist = self.channel_stats['revenue_diversity_score'].value_counts().sort_index()
        axes[1, 0].bar(diversity_dist.index, diversity_dist.values)
        axes[1, 0].set_title('Channel Revenue Diversity')
        axes[1, 0].set_xlabel('Revenue Count')

        outlier_mask = remove_outliers(self.channel_stats['view_count_sum'])
        filtered_data = self.channel_stats[outlier_mask]
        filtered_data['view_count_sum'].replace(0, 1)

        sns.boxenplot(x='revenue_diversity_score', y='view_count_sum', data=filtered_data)
        plt.yscale('log')
        axes[1, 1].set_xlabel('Channel Revenue Diversity')
        axes[1, 1].set_ylabel('View Count')
        axes[1, 1].set_title('Diversity VS View count')

        plt.tight_layout()
        plt.savefig('revenue_analysis.png', dpi=600)
        plt.show()


    def save_results(self):
        merged_channel_data = pd.merge(
            self.channel_stats,
            self.df_channels[['channel', 'name_cc']],
            left_on='channel_id',
            right_on='channel',
            how='inner'
        ).drop(columns='channel_id')
        merged_channel_data.to_csv('channel_revenue_analysis.csv', index=False)

        key_features = ['display_id', 'channel_id', 'title', 'view_count'] + \
                       [f'has_{rt}' for rt in self.revenue_patterns.keys()] + ['num_links']
        self.df_videos[key_features].to_csv('video_revenue_features.csv', index=False)

    def topic_model_strategy(self):
        print("Topic based external strategy analysis")

        topic_model = joblib.load(self.topic_model_path)
        embeddings_cpu = torch.load(self.embeddings_path)
        df = pd.read_parquet(self.money_df_path)
        def only_english_words(text):
            words = text.split()
            filtered_words = [word for word in words if word.lower() in english_words]
            return ' '.join(filtered_words)

        text = df['clean_text'].tolist()
        text_with_stopwords = [
            only_english_words(' '.join([word for word in doc.split() if word.lower() not in stop_words]))
            for doc in text
        ]

        topics, probabilities = topic_model.transform(text_with_stopwords, embeddings=embeddings_cpu)
        self.df_videos['topic'] = topics
        external_videos = self.df_videos[self.df_videos['num_links'] > 0].copy()

        topic_info = topic_model.get_topic_info()
        n_topics = len(topic_info) - 1

        self.topic_revenue_analysis = {}

        for topic_id in topic_info['Topic'].unique():
            if topic_id == -1:
                continue

            topic_videos = external_videos[external_videos['topic'] == topic_id]

            if len(topic_videos) > 0:
                revenue_stats = {}
                for revenue_type in self.revenue_patterns.keys():
                    col_name = f'has_{revenue_type}'
                    if col_name in topic_videos.columns:
                        revenue_stats[revenue_type] = topic_videos[col_name].mean()

                topic_words = topic_model.get_topic(topic_id)
                if topic_words:
                    top_words = [word for word, _ in topic_words[:5]]
                else:
                    top_words = []

                self.topic_revenue_analysis[topic_id] = {
                    'video_count': len(topic_videos),
                    'revenue_methods': revenue_stats,
                    'top_words': top_words,
                    'avg_links': topic_videos['num_links'].mean(),
                    'topic_name': topic_info[topic_info['Topic'] == topic_id]['Name'].values[0] if len(
                        topic_info[topic_info['Topic'] == topic_id]) > 0 else f"Topic {topic_id}"
                }

        if len(external_videos) > 50:
            revenue_features = []
            for revenue_type in self.revenue_patterns.keys():
                col_name = f'has_{revenue_type}'
                if col_name in external_videos.columns:
                    revenue_features.append(external_videos[col_name].values)

            revenue_features.append(external_videos['num_links'].values / external_videos['num_links'].max())
            revenue_matrix = np.array(revenue_features).T

            n_clusters = min(5, len(external_videos) // 100)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            external_videos['revenue_cluster'] = kmeans.fit_predict(revenue_matrix)

            self.revenue_clusters = {}
            for cluster_id in range(n_clusters):
                cluster_videos = external_videos[external_videos['revenue_cluster'] == cluster_id]

                cluster_profile = {}
                for revenue_type in self.revenue_patterns.keys():
                    col_name = f'has_{revenue_type}'
                    if col_name in cluster_videos.columns:
                        cluster_profile[revenue_type] = cluster_videos[col_name].mean()

                top_topics = cluster_videos['topic'].value_counts().head(3)

                self.revenue_clusters[cluster_id] = {
                    'size': len(cluster_videos),
                    'profile': cluster_profile,
                    'avg_links': cluster_videos['num_links'].mean(),
                    'avg_views': cluster_videos['view_count'].mean(),
                    'top_topics': top_topics.to_dict()
                }

            self.external_videos_with_topics = external_videos

    def visualization_topic(self):
        if hasattr(self, 'topic_revenue_analysis') and self.topic_revenue_analysis:
            fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12))

            topic_sizes = [v['video_count'] for v in self.topic_revenue_analysis.values()]
            topic_labels = [v['topic_name'][:20] + '...' if len(v['topic_name']) > 20 else v['topic_name']
                            for v in self.topic_revenue_analysis.values()]

            if len(topic_sizes) > 10:
                sorted_topics = sorted(zip(topic_sizes, topic_labels), reverse=True)[:10]
                topic_sizes, topic_labels = zip(*sorted_topics)
                topic_sizes = list(topic_sizes)
                topic_labels = list(topic_labels)
                other_size = sum([v['video_count'] for v in self.topic_revenue_analysis.values()]) - sum(topic_sizes)
                if other_size > 0:
                    topic_sizes.append(other_size)
                    topic_labels.append('Other topic')

            axes2[0, 0].pie(topic_sizes, labels=topic_labels, autopct='%1.1f%%')
            axes2[0, 0].set_title('External strategy topic distribution')

            top_topics = sorted(self.topic_revenue_analysis.items(),
                                key=lambda x: x[1]['video_count'],
                                reverse=True)[:10]

            topic_revenue_matrix = []
            topic_names = []
            for topic_id, data in top_topics:
                revenue_values = [data['revenue_methods'].get(rt, 0) for rt in self.revenue_patterns.keys()]
                topic_revenue_matrix.append(revenue_values)
                topic_name = data['topic_name'][:15] + '...' if len(data['topic_name']) > 15 else data['topic_name']
                topic_names.append(topic_name)

            topic_revenue_matrix = np.array(topic_revenue_matrix)

            im = axes2[0, 1].imshow(topic_revenue_matrix.T, aspect='auto', cmap='YlOrRd')
            axes2[0, 1].set_yticks(range(len(self.revenue_patterns)))
            axes2[0, 1].set_yticklabels(list(self.revenue_patterns.keys()))
            axes2[0, 1].set_xticks(range(len(topic_names)))
            axes2[0, 1].set_xticklabels(topic_names, rotation=45, ha='right')
            axes2[0, 1].set_title('Topics and Ways to Make Money')
            plt.colorbar(im, ax=axes2[0, 1])

            if hasattr(self, 'revenue_clusters'):
                cluster_sizes = [v['size'] for v in self.revenue_clusters.values()]
                cluster_labels = [f"strategy{i + 1}" for i in range(len(cluster_sizes))]
                axes2[1, 0].bar(cluster_labels, cluster_sizes)
                axes2[1, 0].set_title('External money making strategy cluster result')
                axes2[1, 0].set_ylabel('video numbers')

                cluster_performance = [(k, v['avg_views']) for k, v in self.revenue_clusters.items()]
                cluster_performance.sort(key=lambda x: x[1], reverse=True)

                cluster_ids = [f"strategy{i + 1}" for i, _ in cluster_performance]
                avg_views = [views for _, views in cluster_performance]

                axes2[1, 1].bar(cluster_ids, avg_views)
                axes2[1, 1].set_title('Average views for different strategy clusters')
                axes2[1, 1].set_ylabel('Average view count')
                axes2[1, 1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig('topic_revenue_analysis.png', dpi=600)
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