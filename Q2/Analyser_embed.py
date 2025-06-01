import joblib
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords, words
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import lzma
import json

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('words')
english_words = set(words.words())

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

sample = False

class Analyser:
    def __init__(self):
        self.regression_results = None
        self.style_performance_sub_matrix = None
        self.style_performance_corr = None
        self.correlation_matrix = None
        self.top50_channels = None
        self.top50_videos = None
        self.df = None
        self.regression_results = {}
        self.data_path = "./top_1000_popular_analysis.parquet" if sample else "./popular_analysis.parquet"
        self.topic_model_path = "../Q1/bertopic_model.pkl.lzma"
        self.embeddings_path_cpu = "../Q1/embeddings_cpu.pt"
        self.clean_text_path = "../Q1/money_df_clean_text.parquet"
        self.save_regression_path = "/stacking_model.pkl.lzma"

        self.popular_terms = [
            'free', 'fast', 'easy', 'secret', 'hack', 'trick', 'method',
            'strategy', 'system', 'blueprint', 'guide', 'tutorial',
            'guaranteed', 'proven', 'ultimate', 'best', 'top', 'money',
            'rich', 'wealth', 'million', 'passive', 'income'
        ]
        self.style_features = [
            'title_length', 'title_word_count', 'description_length', 'description_word_count',
            'duration', 'title_has_exclamation', 'title_has_question',
            'title_has_numbers', 'title_has_dollar', 'title_has_caps', 'title_polarity', 'title_subjectivity',
            'title_popular_terms_count', 'primary_topic'
        ]

    def load_data(self):
        self.df = pd.read_parquet(self.data_path)
        print(f'data shape: {self.df.shape}')

    def top_content(self):
        self.top50_videos = self.df.nlargest(100, "view_count")

        channel_stats = self.df.groupby('channel_id').agg({
            'view_count': ['sum', 'mean', 'count'],
            'like_count': 'sum',
            'num_comms': 'sum',
            'engagement_rate': 'mean',
            'like_rate': 'mean'
        }).round(4)
        channel_stats.columns = [f"{col[0]}_{col[1]}" for col in channel_stats.columns]
        channel_stats = channel_stats.reset_index()
        # print(channel_stats)
        self.top50_channels = channel_stats.nlargest(50, 'view_count_sum')
        # print(self.top_channels)

    def content_style(self):
        self.df['title_length'] = self.df['title'].str.len()
        self.df['title_word_count'] = self.df['title'].str.split().str.len()
        self.df['description_length'] = self.df['description'].str.len()
        self.df['description_word_count'] = self.df['description'].str.split().str.len()

        # special chars
        self.df['title_has_exclamation'] = self.df['title'].str.contains('!').astype(int)
        self.df['title_has_question'] = self.df['title'].str.contains(r'\?').astype(int)
        self.df['title_has_numbers'] = self.df['title'].str.contains(r'\d').astype(int)
        self.df['title_has_dollar'] = self.df['title'].str.contains('$').astype(int)
        self.df['title_has_caps'] = self.df['title'].str.contains(r'[A-Z]{2,}').astype(int)


        for term in self.popular_terms:
            self.df[f'title_has_{term}'] = self.df['title'].str.lower().str.contains(term, na=False).astype(int)
        popular_term_columns = [f'title_has_{term}' for term in self.popular_terms]
        self.df['title_popular_terms_count'] = self.df[popular_term_columns].sum(axis=1)
        self.df = self.df.drop(columns=popular_term_columns)

        # title sentiment
        def get_sentiment(text):
            if pd.isna(text) or text == '':
                return 0, 0
            else:
                blob = TextBlob(text)
                return blob.sentiment  # (polarity, subjectivity)

        self.df[['title_polarity', 'title_subjectivity']] = self.df['title'].apply(lambda x: pd.Series(get_sentiment(x)))

        # print(self.df.head(100))

    def content_topic(self):
        def load_bertopic():
            topic_model = joblib.load(self.topic_model_path)
            embeddings_cpu = torch.load(self.embeddings_path_cpu)
            clean_text_df = pd.read_parquet(self.clean_text_path)

            def only_english_words(text):
                words = text.split()
                filtered_words = [word for word in words if word.lower() in english_words]
                return ' '.join(filtered_words)

            text = clean_text_df['clean_text'].tolist()
            self.text_with_stopwords = [
                only_english_words(' '.join([word for word in doc.split() if word.lower() not in stop_words]))
                for doc in text
            ]
            topics, probabilities = topic_model.transform(text_with_stopwords, embeddings=embeddings_cpu)
            topic_info = topic_model.get_topic_info()
            print(f"Discovered {len(topic_info)} topics, the first 5 topics are:")
            print(topic_info.head(5))
            return text_with_stopwords, topics, probabilities, topic_model, topic_info

        text_with_stopwords, topics, probabilities, topic_model, topic_info = load_bertopic()

        self.df['primary_topic'] = topics
        self.df['topic_confidence'] = 0.0
        for i, prob in enumerate(probabilities):
            if isinstance(prob, (list, tuple)) and prob:
                self.df.loc[i, 'topic_confidence'] = max(prob)
            elif not isinstance(prob, (list, tuple)):
                self.df.loc[i, 'topic_confidence'] = prob

        print(self.df.head(5))


        topics_keywords = []
        topic_labels = []

        for topic_id in sorted(self.df['primary_topic'].unique()):
            if topic_id == -1:
                continue
            topic_words = topic_model.get_topic(topic_id)

            keywords = [word for word, score in topic_words[:10]]
            topics_keywords.append(', '.join(keywords))
            topic_label = f"Topic {topic_id}: {', '.join(keywords[:3])}"
            topic_labels.append(topic_label)
        print(topics_keywords[:10],topic_labels[:10])

    def correlation_analysis(self):
        # self.style_features.extend([f'title_has_{term}' for term in self.popular_terms])
        performance_metrics = ['view_count', 'like_count', 'num_comms', 'engagement_rate', 'like_rate']
        all_columns = self.style_features + performance_metrics

        self.correlation_matrix = self.df[all_columns].corr()
        self.style_performance_sub_matrix = self.correlation_matrix.loc[self.style_features, performance_metrics]
        print(self.style_performance_sub_matrix)

    def regression_analysis(self):
        X = self.df[self.style_features].fillna(0)
        self.df['log_views'] = np.log1p(self.df['view_count'])
        self.df['log_likes'] = np.log1p(self.df['like_count'])
        self.df['log_comments'] = np.log1p(self.df['num_comms'])

        Q1_views = self.df['view_count'].quantile(0.25)
        Q3_views = self.df['view_count'].quantile(0.75)
        IQR_views = Q3_views - Q1_views
        lower_bound = Q1_views - 1.5 * IQR_views
        upper_bound = Q3_views + 1.5 * IQR_views
        mask = (self.df['view_count'] >= lower_bound) & (self.df['view_count'] <= upper_bound)
        X_clean = X[mask]
        print(f"Lines after IQR: {len(X_clean)}")

        targets = [
            ('log_views', 'view_nums'),
            ('log_likes', 'like_nums'),
            ('log_comments', 'comment_nums')
        ]

        print("using Stacking model")
        for target_col, target_name in targets:
            print("training: " + target_name + "stack model")
            y_clean = self.df.loc[mask, target_col]
            X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=0)

            base_models = [
                ('rf', RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)),
                ('xgb', xgb.XGBRegressor(n_estimators=1000, random_state=0, n_jobs=-1)),
                ('lgb', lgb.LGBMRegressor(n_estimators=1000, random_state=0, n_jobs=-1))
            ]
            meta_model = Ridge(random_state=0)
            stacking_model = StackingRegressor(
                estimators=base_models,
                final_estimator=meta_model,
                cv=5,
                n_jobs=-1
            )
            stacking_model.fit(X_train, y_train)

            y_pred = stacking_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            print(f"stacking model - R²: {r2:.4f}, MSE: {mse:.4f}")

            perm_importance = permutation_importance(
                stacking_model, X_test, y_test,
                n_repeats=10, random_state=0, n_jobs=-1
            )
            feature_importance = pd.DataFrame({
                'feature': self.style_features,
                'importance': perm_importance.importances_mean
            }).sort_values('importance', ascending=False)
            print(feature_importance)

            self.regression_results[target_col] = {
                'target_name': target_name,
                'r2': r2,
                'mse': mse,
                'feature_importance': feature_importance
            }

            with lzma.open(target_name+self.save_regression_path, 'wb') as f:
                joblib.dump(stacking_model, f)


if __name__ == '__main__':
    analyser = Analyser()
    analyser.load_data()
    analyser.top_content()
    print("title style analysis by textblob")
    analyser.content_style()
    print("topic model by bertopic")
    analyser.content_topic()
    print("correlation analysis")
    analyser.correlation_analysis()
    print("regression analysis")
    analyser.regression_analysis()
