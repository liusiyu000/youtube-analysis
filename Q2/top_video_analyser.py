import lzma
import os

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xgboost as xgb
from joblib import Parallel, delayed
from nltk.corpus import stopwords, words
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from textblob import TextBlob
from content_style import add_style_features, STYLE_LABELS
from catboost import CatBoostRegressor

if os.path.exists("C:/temp"):
    os.environ['JOBLIB_TEMP_FOLDER'] = "C:/temp"

nltk.download('stopwords')
nltk.download('vader_lexicon')
stop_words = set(stopwords.words('english'))
nltk.download('words')
english_words = set(words.words())

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

SAMPLE = False
train_single = False

class Analyser:
    def __init__(self):
        self.top100_videos = None
        self.regression_results = None
        self.style_performance_sub_matrix = None
        self.style_performance_corr = None
        self.correlation_matrix = None
        self.top50_channels = None
        self.top50_videos = None
        self.df = None
        self.regression_results = {}
        self.data_path = "./top_1000_popular_analysis.parquet" if SAMPLE else "./popular_analysis.parquet"
        self.topic_model_path = "../Q1/bertopic_model.pkl.lzma"
        self.embeddings_path_cpu = "../Q1/embeddings_cpu.pt"
        self.clean_text_path = "../Q1/money_df_clean_text.parquet"
        self.save_regression_path = "_stacking_model.pkl.lzma"


        self.popular_terms = [
            'free', 'fast', 'easy', 'secret', 'hack', 'trick', 'method',
            'strategy', 'system', 'blueprint', 'guide', 'tutorial',
            'guaranteed', 'proven', 'ultimate', 'best', 'top', 'money',
            'rich', 'wealth', 'million', 'passive', 'income'
        ]

        self.basic_features = [
            'title_length', 'title_word_count', 'description_length', 'description_word_count',
            'duration', 'title_has_exclamation', 'title_has_question',
            'title_has_numbers', 'title_has_dollar', 'title_has_caps',
            'title_popular_terms_count'
        ]

        self.sentiment_features = [
            'title_polarity', 'title_subjectivity',
            'title_sentiment_compound', 'title_sentiment_pos',
            'title_sentiment_neg', 'title_sentiment_neu',
            'description_polarity', 'description_subjectivity',
            'description_sentiment_compound'
        ]

        self.time_features = [
            'publish_weekday', 'days_since_publish'
        ]

        self.channel_features = [
            'view_count_mean', 'like_count_mean', 'num_comms_mean',
            'view_count_count','view_consistency'
        ]

        self.topic_features = ['primary_topic', 'topic_confidence']

        self.style_features = []

    def load_data(self):
        self.df = pd.read_parquet(self.data_path)
        print(f'data shape: {self.df.shape}')
        print(self.df.head(5))

        self.df['publish_date'] = pd.to_datetime(self.df['upload_date'], format='mixed')
        self.df['publish_hour'] = self.df['publish_date'].dt.hour
        self.df['publish_day'] = self.df['publish_date'].dt.day
        self.df['publish_month'] = self.df['publish_date'].dt.month
        self.df['publish_year'] = self.df['publish_date'].dt.year
        self.df['publish_weekday'] = self.df['publish_date'].dt.weekday

        self.df['crawl_date'] = pd.to_datetime(self.df['crawl_date'], format='mixed')
        self.df['days_since_publish'] = (self.df['crawl_date'] - self.df['publish_date']).dt.days

    def top_content(self):
        self.top50_videos = self.df.nlargest(100, "view_count")

        channel_stats = self.df.groupby('channel_id').agg({
            'view_count': ['sum', 'mean', 'std', 'count'],
            'like_count': ['mean', 'std'],
            'num_comms': ['mean', 'std'],
            'engagement_rate': 'mean',
            'like_rate': 'mean'
        }).round(4)

        channel_stats.columns = [f"{col[0]}_{col[1]}" for col in channel_stats.columns]
        channel_stats = channel_stats.reset_index()
        channel_stats['view_consistency'] = np.where(
            channel_stats['view_count_std'] > 0,
            channel_stats['view_count_mean'] / channel_stats['view_count_std'],
            0
        )
        self.df = self.df.merge(
            channel_stats[['channel_id', 'view_count_mean', 'view_count_std', 'view_count_count',
                          'like_count_mean', 'num_comms_mean', 'view_consistency']],
            on='channel_id',
            how='left'
        )
        # print(channel_stats)
        self.top50_channels = channel_stats.nlargest(50, 'view_count_sum')
        self.top100_videos = self.df.nlargest(100, "view_count")
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
        self.df = add_style_features(self.df)

        def avg_word_length(text):
            if pd.isna(text) or text == '':
                return 0
            words = text.split()
            if not words:
                return 0
            return sum(len(word) for word in words) / len(words)

        self.df['title_avg_word_length'] = self.df['title'].apply(avg_word_length)

        for term in self.popular_terms:
            self.df[f'title_has_{term}'] = self.df['title'].str.lower().str.contains(term, na=False).astype(int)
        popular_term_columns = [f'title_has_{term}' for term in self.popular_terms]
        self.df['title_popular_terms_count'] = self.df[popular_term_columns].sum(axis=1)
        self.df = self.df.drop(columns=popular_term_columns)

        sid = SentimentIntensityAnalyzer()

        def enhanced_sentiment(text):
            if pd.isna(text) or text == '':
                return 0, 0, 0, 0, 0, 0
            else:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                sentiment_scores = sid.polarity_scores(text)
                compound = sentiment_scores['compound']
                pos = sentiment_scores['pos']
                neg = sentiment_scores['neg']
                neu = sentiment_scores['neu']

                return polarity, subjectivity, compound, pos, neg, neu

        results = Parallel(n_jobs=-1)(
            delayed(enhanced_sentiment)(text) for text in self.df['title']
        )

        sentiment_results = pd.DataFrame(results, columns=['polarity', 'subjectivity', 'compound', 'pos', 'neg', 'neu'])

        self.df['title_polarity'] = sentiment_results['polarity']
        self.df['title_subjectivity'] = sentiment_results['subjectivity']
        self.df['title_sentiment_compound'] = sentiment_results['compound']
        self.df['title_sentiment_pos'] = sentiment_results['pos']
        self.df['title_sentiment_neg'] = sentiment_results['neg']
        self.df['title_sentiment_neu'] = sentiment_results['neu']

        results = Parallel(n_jobs=-1)(
            delayed(enhanced_sentiment)(text) for text in self.df['description']
        )
        description_sentiment = pd.DataFrame(
            [res[:3] for res in results],
            columns=['polarity', 'subjectivity', 'compound']
        )

        self.df['description_polarity'] = description_sentiment['polarity']
        self.df['description_subjectivity'] = description_sentiment['subjectivity']
        self.df['description_sentiment_compound'] = description_sentiment['compound']

        self.df['title_sentiment_magnitude'] = self.df['title_polarity'].abs()
        self.df['title_emotion_strength'] = self.df['title_sentiment_magnitude'] * self.df['title_subjectivity']

        self.df['duration_seconds'] = self.df['duration']
        self.df['duration_minutes'] = self.df['duration'] / 60
        self.df['duration_category'] = pd.cut(
            self.df['duration_minutes'],
            bins=[0, 3, 10, 20, 60, float('inf')],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

        self.df['duration_log'] = np.log1p(self.df['duration'])
        self.df['duration_sqrt'] = np.sqrt(self.df['duration'])
        # print(self.df.head(100))

    def content_topic(self):
        topic_model = joblib.load(self.topic_model_path)
        embeddings_cpu = torch.load(self.embeddings_path_cpu)
        clean_text_df = pd.read_parquet(self.clean_text_path)
        def only_english_words(text):
            words = text.split()
            filtered_words = [word for word in words if word.lower() in english_words]
            return ' '.join(filtered_words)

        text = clean_text_df['clean_text'].tolist()
        text_with_stopwords = [
            only_english_words(' '.join([word for word in doc.split() if word.lower() not in stop_words]))
            for doc in text
        ]
        topics, probabilities = topic_model.transform(text_with_stopwords, embeddings=embeddings_cpu)
        topic_info = topic_model.get_topic_info()
        print(f"Discovered {len(topic_info)} topics, the first 5 topics are:")
        print(topic_info.head(5))

        self.df['primary_topic'] = topics
        self.df['topic_confidence'] = 0.0

        max_topic_id = max([max(p.keys()) if isinstance(p, dict) else -1 for p in probabilities if p])
        num_topics = max_topic_id + 1
        topic_prob_matrix = np.zeros((len(self.df), num_topics))

        for i, prob_dist in enumerate(probabilities):
            if isinstance(prob_dist, (list, tuple)) and prob_dist:
                max_prob = max(prob_dist)
                self.df.loc[i, 'topic_confidence'] = max_prob

                for topic_idx, prob in enumerate(prob_dist):
                    topic_prob_matrix[i, topic_idx] = prob
            elif not isinstance(prob_dist, (list, tuple)):
                self.df.loc[i, 'topic_confidence'] = prob_dist


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
        print(f"topic keywords example: {topics_keywords[:3]}")

        topic_performance = self.df.groupby('primary_topic').agg({
            'view_count': 'mean',
            'like_count': 'mean'
        }).reset_index()

        self.df = self.df.merge(
            topic_performance,
            on='primary_topic',
            how='left',
            suffixes=('', '_topic_avg')
        )

        self.df['view_vs_topic_avg'] = self.df['view_count'] / self.df['view_count_topic_avg']
        self.df['like_vs_topic_avg'] = self.df['like_count'] / self.df['like_count_topic_avg']

        self.df['view_vs_topic_avg'].replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df['like_vs_topic_avg'].replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df['view_vs_topic_avg'].fillna(1, inplace=True)
        self.df['like_vs_topic_avg'].fillna(1, inplace=True)

        self.topic_features.extend(['view_count_topic_avg', 'like_count_topic_avg',
                                    'view_vs_topic_avg', 'like_vs_topic_avg'])

    def select_features(self):
        self.style_features = self.basic_features + self.sentiment_features + self.time_features + self.channel_features + ['title_avg_word_length', 'title_sentiment_magnitude',
                 'title_emotion_strength']

        self.style_features += [f"style_{lbl}" for lbl in STYLE_LABELS]

        if not SAMPLE:
            self.style_features += self.topic_features

    def correlation_analysis(self):
        # self.style_features.extend([f'title_has_{term}' for term in self.popular_terms])
        performance_metrics = ['view_count', 'like_count', 'num_comms', 'engagement_rate', 'like_rate']
        exclude_features = [
            'view_count_mean', 'view_count_count', 'view_consistency',
                           'view_count_topic_avg', 'view_vs_topic_avg',
            'like_count_mean', 'like_rate', 'like_count_topic_avg',
                           'like_vs_topic_avg',
            'num_comms_mean',
            'engagement_rate', 'like_rate', 'like_count_mean',
                                'num_comms_mean',
            'like_rate', 'like_count_mean', 'engagement_rate',
                          'like_count_topic_avg', 'like_vs_topic_avg'
        ]

        self.style_performance_corr = {}
        for metric in performance_metrics:
            features_for_metric = [f for f in self.style_features
                                   if f not in exclude_features]
            available_features = [f for f in features_for_metric if f in self.df.columns]
            if metric in self.df.columns and available_features:
                metric_corr = self.df[available_features + [metric]].corr()
                self.style_performance_corr[metric] = metric_corr[metric].drop(metric)
        all_features = list(set(self.style_features))
        all_columns = all_features + performance_metrics
        all_columns = [col for col in all_columns if col in self.df.columns]

        self.correlation_matrix = self.df[all_columns].corr()
        self.style_performance_sub_matrix = self.correlation_matrix.loc[self.style_features, performance_metrics].copy()

        for metric in performance_metrics:
            if metric in self.style_performance_sub_matrix.columns:
                for feature in exclude_features:
                    if feature in self.style_performance_sub_matrix.index:
                        self.style_performance_sub_matrix.loc[feature, metric] = np.nan


    def regression_analysis(self):
        import gc
        X = self.df[self.style_features].copy()
        X = X.fillna(X.median())

        if not os.path.exists("video_features_all.parquet"):
            self.df['log_views'] = np.log1p(self.df['view_count'])
            self.df['log_likes'] = np.log1p(self.df['like_count'])
            self.df['log_comments'] = np.log1p(self.df['num_comms'])

            self.df.to_parquet("./video_features_all.parquet", compression="snappy")
            print("saved all features")



        targets = [
            ('log_views', 'view_nums'),
            ('log_likes', 'like_nums'),
            ('log_comments', 'comment_nums')
        ]

        exclude_features = [
            'view_count_mean', 'view_count_count', 'view_consistency',
                          'view_count_topic_avg', 'view_vs_topic_avg',
            'like_count_mean', 'like_rate', 'like_count_topic_avg',
                          'like_vs_topic_avg', 'engagement_rate',
            'num_comms_mean', 'engagement_rate'
        ]

        for target_col, target_name in targets:
            print("training: " + target_name + " stack model")
            current_features = [f for f in self.style_features if f not in exclude_features]
            current_features = [f for f in current_features if f in X.columns]
            X_filtered = X[current_features].copy()

            flag = False
            meta_model_names = ['ridge', 'elastic', 'lasso']
            for meta_name in meta_model_names:
                model_path = f"./{target_name}_{meta_name}_{int(0.95 * 100)}{self.save_regression_path}"
                if os.path.exists(model_path):
                    flag = True
                    break
            if flag == True:
                with lzma.open(model_path, 'rb') as f:
                    existing_model = joblib.load(f)
                print(f"Found existing model: {model_path}, R²: {existing_model.get('r2', 0)}")

                final_stacking_model = existing_model['model']
                scaler = existing_model['scaler']
                power_transformer = existing_model['transformer']
                r2 = existing_model['r2']
                mse = existing_model.get('mse', 0)
                meta_model = existing_model['meta_model']
                feature_importance = existing_model['feature_importance']

                self.regression_results[target_col] = {
                    'target_name': target_name,
                    'r2': r2,
                    'mse': mse,
                    'feature_importance': feature_importance,
                    'selected_features': current_features,
                    'meta_model': meta_model,
                    'quantile': 0.95,
                    'excluded_features': exclude_features
                }
                # print("\nMost important 10 features:")
                # print(feature_importance.head(10))

            else:
                upper_q = self.df[target_col].quantile(0.95)
                mask = self.df[target_col] <= upper_q
                X_filtered_masked = X_filtered[mask]
                y_filtered = self.df.loc[mask, target_col]

                X_train, X_test, y_train, y_test = train_test_split(X_filtered_masked, y_filtered, test_size=0.3,
                                                                    random_state=0)

                scaler = StandardScaler()
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )

                power_transformer = PowerTransformer(method='yeo-johnson')
                y_train_transformed = power_transformer.fit_transform(y_train.values.reshape(-1, 1)).ravel()

                device = "cuda" if torch.cuda.is_available() else "cpu"
                task_type = "GPU" if torch.cuda.is_available() and train_single else "CPU"
                base_models = [
                    ('cbr',
                     CatBoostRegressor(iterations=300, depth=12, learning_rate=0.05, devices=device, random_seed=0, thread_count=-1, verbose=False, task_type=task_type)),
                    ('rf',
                     RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=5, random_state=0, n_jobs=-1)),
                    ('xgb', xgb.XGBRegressor(n_estimators=500, max_depth=10, learning_rate=0.05, tree_method='auto',
                                             subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1,device=device)),
                    ('lgb', lgb.LGBMRegressor(n_estimators=500, max_depth=10, learning_rate=0.05,
                                              subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1, verbose=-1)),
                    ('gbr', GradientBoostingRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=0)),
                ]

                meta_models = [
                    ('ridge', Ridge(alpha=1.0, random_state=0)),
                    ('elastic', ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=0)),
                    ('lasso', Lasso(alpha=0.1, random_state=0))
                ]

                if train_single == True:
                    print("\nEvaluating single base models:")
                    for name, model in base_models:
                        model.fit(X_train_scaled, y_train_transformed)
                        y_pred_transformed = model.predict(X_test_scaled)
                        y_pred = power_transformer.inverse_transform(y_pred_transformed.reshape(-1, 1)).ravel()

                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        print(f"  {name.upper()} - R²: {r2:.4f}, MSE: {mse:.4f}")

                else:
                    best_meta_model = None
                    best_r2 = -float('inf')
                    best_mse = float('inf')

                    for meta_name, meta_model in meta_models:
                        print(f"Using {meta_name} meta model ...")

                        stacking_model = StackingRegressor(
                            estimators=base_models,
                            final_estimator=meta_model,
                            # cv=5,
                            n_jobs=-1
                        )

                        stacking_model.fit(X_train_scaled, y_train_transformed)
                        y_test_pred_transformed = stacking_model.predict(X_test_scaled)
                        y_test_pred = power_transformer.inverse_transform(y_test_pred_transformed.reshape(-1, 1)).ravel()

                        r2 = r2_score(y_test, y_test_pred)
                        mse = mean_squared_error(y_test, y_test_pred)
                        print(f"  {meta_name} - R²: {r2:.4f}, MSE: {mse:.4f}")

                        if r2 > best_r2:
                            best_r2 = r2
                            best_mse = mse
                            best_meta_model = meta_name

                    print(f"Best meta model: {best_meta_model}, R²: {best_r2:.4f}, MSE: {best_mse:.4f}")

                    final_meta_model = None
                    for name, model in meta_models:
                        if name == best_meta_model:
                            final_meta_model = model

                    final_stacking_model = StackingRegressor(
                        estimators=base_models,
                        final_estimator=final_meta_model,
                        # cv=5,
                        n_jobs=-1
                    )

                    X_all_scaled = pd.DataFrame(
                        scaler.fit_transform(X_filtered_masked),
                        columns=X_filtered_masked.columns,
                        index=X_filtered_masked.index
                    )
                    y_all_transformed = power_transformer.fit_transform(y_filtered.values.reshape(-1, 1)).ravel()
                    final_stacking_model.fit(X_all_scaled, y_all_transformed)

                    perm_importance = permutation_importance(
                        final_stacking_model, X_test_scaled, y_test,
                        n_repeats=1, random_state=0, n_jobs=-1
                    )

                    feature_importance = pd.DataFrame({
                        'feature': current_features,
                        'importance': perm_importance.importances_mean
                    }).sort_values('importance', ascending=False)

                    print("\nMost important 10 features:")
                    print(feature_importance.head(10))

                    self.regression_results[target_col] = {
                        'target_name': target_name,
                        'r2': best_r2,
                        'mse': best_mse,
                        'feature_importance': feature_importance,
                        'selected_features': current_features,
                        'meta_model': best_meta_model,
                        'quantile': 0.95,
                        'excluded_features': exclude_features
                    }

                    with lzma.open(f"./{target_name}_{best_meta_model}_{int(0.95 * 100)}{self.save_regression_path}", 'wb') as f:
                        model_package = {
                            'model': final_stacking_model,
                            'scaler': scaler,
                            'transformer': power_transformer,
                            'selected_features': current_features,
                            'r2': best_r2,
                            'meta_model': best_meta_model,
                            'feature_importance': feature_importance,
                            'excluded_features': exclude_features,
                            'mse': best_mse,
                        }
                        joblib.dump(model_package, f, compress=('xz', 9), protocol=5)


    def identify_differential_features(self, threshold=0.03):

        df_corr = self.style_performance_sub_matrix.copy()

        df_corr_clean = df_corr.fillna(0)

        differential_features = []
        feature_differences = {}

        for feature in df_corr_clean.index:
            correlations = df_corr_clean.loc[feature].values

            std_corr = np.std(correlations)

            range_corr = np.max(correlations) - np.min(correlations)

            feature_differences[feature] = {
                'std': std_corr,
                'range': range_corr,
                'correlations': correlations
            }

            if std_corr > threshold or range_corr > 0.05:
                differential_features.append(feature)

        return differential_features, feature_differences

    def visualization(self):
        def top_100():
            print("1. Top 100 videos perform distribution")
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Top 100 videos perform distribution', fontsize=16, fontweight='bold')

            axes[0, 0].hist(self.top100_videos['view_count'], bins=30, alpha=0.7,
                            color='steelblue', edgecolor='black', linewidth=0.5)
            axes[0, 0].set_title('View number distribution', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('view number')
            axes[0, 0].set_ylabel('frequency')
            axes[0, 0].ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].hist(self.top100_videos['like_count'], bins=30, alpha=0.7,
                            color='green', edgecolor='black', linewidth=0.5)
            axes[0, 1].set_title('like number distribution', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('like number')
            axes[0, 1].set_ylabel('frequency')
            axes[0, 1].grid(True, alpha=0.3)

            axes[0, 2].hist(self.top100_videos['num_comms'], bins=30, alpha=0.7,
                            color='orange', edgecolor='black', linewidth=0.5)
            axes[0, 2].set_title('comment number distribution', fontsize=12, fontweight='bold')
            axes[0, 2].set_xlabel('comment number')
            axes[0, 2].set_ylabel('frequency')
            axes[0, 2].grid(True, alpha=0.3)

            axes[1, 0].hist(self.top100_videos['engagement_rate'], bins=30, alpha=0.7,
                            color='purple', edgecolor='black', linewidth=0.5)
            axes[1, 0].set_title('engagement rate distribution', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('engagement rate')
            axes[1, 0].set_ylabel('frequency')
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].hist(self.top100_videos['like_rate'], bins=30, alpha=0.7,
                            color='red', edgecolor='black', linewidth=0.5)
            axes[1, 1].set_title('like rate distribution', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('like rate')
            axes[1, 1].set_ylabel('frequency')
            axes[1, 1].grid(True, alpha=0.3)

            duration_minutes = self.top100_videos['duration'] / 60
            axes[1, 2].hist(duration_minutes, bins=30, alpha=0.7,
                            color='brown', edgecolor='black', linewidth=0.5)
            axes[1, 2].set_title('duration distribution', fontsize=12, fontweight='bold')
            axes[1, 2].set_xlabel('duration (minute)')
            axes[1, 2].set_ylabel('frequency')
            axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('top100_videos_distribution.png', dpi=600)
            plt.show()

        def style_performance():
            print("2. Style performance heat map per column...")

            differential_features, feature_differences = self.identify_differential_features(threshold=0.02)

            # df_corr = self.style_performance_sub_matrix
            # metrics = df_corr.columns
            # n_metrics = len(metrics)
            #
            # fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 24), sharey=True)
            #
            # for i, metric in enumerate(metrics):
            #     data = df_corr[[metric]]
            #     vmin = data.min().min()
            #     vmax = data.max().max()
            #     ax = axes[i]
            #
            #     sns.heatmap(data, annot=True, cmap='RdBu_r', center=0, fmt='.3f',
            #                 vmin=vmin, vmax=vmax, cbar=True, ax=ax,
            #                 linewidths=0.5, square=False)
            #
            #     y_labels = ax.get_yticklabels()
            #     for j, label in enumerate(y_labels):
            #         if label.get_text() in differential_features:
            #             label.set_weight('bold')
            #             label.set_color('red')
            #             label.set_fontsize(13)
            #
            #     ax.set_xlabel('')
            #     ax.tick_params(axis='x', labelsize=15)
            #     if i == 0:
            #         ax.set_ylabel('Content Style', fontsize=15)
            #         ax.tick_params(axis='y', labelsize=15)
            #     else:
            #         ax.set_ylabel('')
            #
            # plt.suptitle('Correlation between content style and performance metrics\n',
            #              fontsize=18, fontweight='bold')
            # plt.tight_layout()
            # plt.savefig('feature_correlation_per_column.png', dpi=600)
            # plt.show()

            df_corr = self.style_performance_sub_matrix
            metrics = df_corr.columns
            n_metrics = len(metrics)

            fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 24), sharey=True)

            for i, metric in enumerate(metrics):
                data = df_corr[[metric]]
                vmin = data.min().min()
                vmax = data.max().max()
                ax = axes[i]

                sns.heatmap(data, annot=True, cmap='RdBu_r', center=0, fmt='.3f',
                            vmin=vmin, vmax=vmax, cbar=True, ax=ax,
                            linewidths=0.5, square=False)

                y_labels = []
                for label in ax.get_yticklabels():
                    feature_name = label.get_text()
                    if feature_name in differential_features:
                        new_label = ax.text(
                            label.get_position()[0],
                            label.get_position()[1],
                            feature_name,
                            fontsize=13,
                            fontweight='bold',
                            color='red',
                            ha='right',
                            va='center'
                        )
                        label.set_visible(False)  # 隐藏原标签
                    else:
                        label.set_fontsize(12)
                        label.set_color('black')

                ax.set_xlabel('')
                ax.tick_params(axis='x', labelsize=15)
                if i == 0:
                    ax.set_ylabel('Content style', fontsize=15)
                else:
                    ax.set_ylabel('')

            plt.suptitle('Correlation between content style and performance metrics',
                         fontsize=18, fontweight='bold')
            plt.tight_layout()

            plt.savefig('feature_correlation_per_column.png', dpi=600, bbox_inches='tight')
            plt.show()

        def feature_imp():
            print("3. Feature importance compare")
            n_models = len(self.regression_results)
            fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 12))

            for i, (target, results) in enumerate(self.regression_results.items()):
                importance_df = results['feature_importance'].head(15)

                colors = plt.cm.viridis(importance_df['importance'] / importance_df['importance'].max())
                bars = axes[i].barh(range(len(importance_df)), importance_df['importance'],
                                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                for j, (idx, row) in enumerate(importance_df.iterrows()):
                    axes[i].text(row['importance'] + max(importance_df['importance']) * 0.01, j,
                                 f'{row["importance"]:.3f}', va='center', fontsize=9)

                axes[i].set_yticks(range(len(importance_df)))
                axes[i].set_yticklabels(importance_df['feature'], fontsize=10)
                axes[i].set_xlabel('feature importance', fontsize=12)
                axes[i].set_title(f'{results["target_name"]} predict model\n'
                                  f'(R² = {results["r2"]:.4f}, MSE = {results["mse"]:.4f})',
                                  fontsize=12, fontweight='bold')
                axes[i].grid(axis='x', alpha=0.3)
                axes[i].invert_yaxis()

            plt.suptitle('Feature importance compare', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('feature_importance_comparison.png', dpi=600)
            plt.show()

        def title_impact():
            print("4. Title feature impact")
            title_features = [
                ('title_has_exclamation', '"!"'),
                ('title_has_question', '"?"'),
                ('title_has_numbers', '"number"'),
                ('title_has_dollar', '"dollar"'),
                ('title_has_caps', '"CAP"')
            ]

            popular_features = [(f'title_has_{term}', f'"{term.title()}"') for term in self.popular_terms[:8]]

            available_features = []
            for feature, name in title_features + popular_features:
                if feature in self.df.columns:
                    available_features.append((feature, name))

            if available_features:
                n_features = len(available_features)
                n_cols = 3
                n_rows = (n_features + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
                if n_rows == 1:
                    axes = axes.reshape(1, -1)
                axes = axes.flatten()

                for i, (feature, name) in enumerate(available_features):
                    groups = self.df.groupby(feature)['view_count'].mean()
                    if len(groups) > 1:
                        improvement = (groups.iloc[1] / groups.iloc[0] - 1) * 100
                        usage_rate = self.df[feature].mean() * 100

                        bars = axes[i].bar([0, 1], groups.values,
                                           color=['lightcoral', 'lightblue'],
                                           alpha=0.8, edgecolor='black', linewidth=1)

                        for bar, value in zip(bars, groups.values):
                            axes[i].text(bar.get_x() + bar.get_width() / 2,
                                         bar.get_height() + bar.get_height() * 0.02,
                                         f'{value:.0f}', ha='center', va='bottom', fontsize=12)

                        axes[i].set_title(f'Impact of {name}\nusage_rate: {usage_rate:.1f}%, '
                                          f'Increased: {improvement:+.1f}%',
                                          fontsize=12, fontweight='bold')
                        axes[i].set_xlabel('Include or not')
                        axes[i].set_ylabel('Average view count')
                        axes[i].set_xticks([0, 1])
                        axes[i].set_xticklabels(['No', 'Yes'])
                        axes[i].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
                        axes[i].grid(axis='y', alpha=0.3)

                for i in range(len(available_features), len(axes)):
                    axes[i].set_visible(False)

                plt.suptitle('Effect of title features on video views', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig('title_features_impact.png', dpi=600)
                plt.show()

        def topic_analysis():
            print("5. Topic analysis")

            topic_stats = self.df.groupby('primary_topic').agg({
                'view_count': ['mean', 'count'],
                'like_count': 'mean',
                'num_comms': 'mean',
                'engagement_rate': 'mean',
                'like_rate': 'mean',
                'duration': 'mean'
            }).round(2)

            topic_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0]
                                   for col in topic_stats.columns]
            topic_stats = topic_stats.reset_index()

            valid_topics = topic_stats[topic_stats['view_count_count'] >= 10]
            valid_topics = valid_topics[valid_topics['primary_topic'] != -1]

            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            fig.suptitle('Compare of performance of different topic', fontsize=16, fontweight='bold')

            colors = plt.cm.tab20(np.linspace(0, 1, 10))

            top_performance = valid_topics.nlargest(20, 'view_count_mean')
            selected_topics = top_performance.drop_duplicates('primary_topic').head(10)

            bars1 = axes[0, 0].bar(range(len(selected_topics)), selected_topics['view_count_mean'],
                                   color=colors,
                                   alpha=0.8, edgecolor='black')
            axes[0, 0].set_title('Average count of views', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Topic ID')
            axes[0, 0].set_ylabel('Average views')
            axes[0, 0].set_xticks(range(10))
            axes[0, 0].set_xticklabels([f'T{int(topic)}' for topic in selected_topics['primary_topic']], rotation=45)
            axes[0, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
            axes[0, 0].grid(axis='y', alpha=0.3)

            top_performance = valid_topics.nlargest(20, 'engagement_rate_mean')
            selected_topics = top_performance.drop_duplicates('primary_topic').head(10)

            bars2 = axes[0, 1].bar(range(len(selected_topics)), selected_topics['engagement_rate_mean'],
                                   color=colors,
                                   alpha=0.8, edgecolor='black')
            axes[0, 1].set_title('Average rate of engagement', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Topic ID')
            axes[0, 1].set_ylabel('Average engagement')
            axes[0, 1].set_xticks(range(10))
            axes[0, 1].set_xticklabels([f'T{int(topic)}' for topic in selected_topics['primary_topic']], rotation=45)
            axes[0, 1].grid(axis='y', alpha=0.3)

            top_performance = valid_topics.nlargest(20, 'like_count_mean')
            selected_topics = top_performance.drop_duplicates('primary_topic').head(10)

            bars3 = axes[1, 0].bar(range(len(selected_topics)), selected_topics['like_count_mean'],
                                   color=colors,
                                   alpha=0.8, edgecolor='black')
            axes[1, 0].set_title('Average count of likes', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Topics')
            axes[1, 0].set_ylabel('Average likes')
            axes[1, 0].set_xticks(range(10))
            axes[1, 0].set_xticklabels([f'T{int(topic)}' for topic in selected_topics['primary_topic']], rotation=45)
            axes[1, 0].grid(axis='y', alpha=0.3)

            top_performance = valid_topics.nlargest(20, 'view_count_count')
            selected_topics = top_performance.drop_duplicates('primary_topic').head(10)

            bars4 = axes[1, 1].bar(range(len(selected_topics)), selected_topics['view_count_count'],
                                   color=colors,
                                   alpha=0.8, edgecolor='black')
            axes[1, 1].set_title('Video number of topic', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Topic ID')
            axes[1, 1].set_ylabel('Video numbers')
            axes[1, 1].set_xticks(range(10))
            axes[1, 1].set_xticklabels([f'T{int(topic)}' for topic in selected_topics['primary_topic']], rotation=45)
            axes[1, 1].grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig('topic_performance_analysis.png', dpi=600)
            plt.show()

            # ============================================
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            top_by_views = valid_topics.nlargest(15, 'view_count_mean')
            ax1.barh(range(len(top_by_views)), top_by_views['view_count_mean'],
                     color=plt.cm.viridis(np.linspace(0, 1, len(top_by_views))), alpha=0.8)
            ax1.set_yticks(range(len(top_by_views)))
            ax1.set_yticklabels([f'Topic {int(topic)}' for topic in top_by_views['primary_topic']])
            ax1.set_xlabel('Average view count')
            ax1.set_title('Top 15 view topics', fontsize=12, fontweight='bold')
            ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
            ax1.grid(axis='x', alpha=0.3)
            ax1.invert_yaxis()

            top_by_count = valid_topics.nlargest(15, 'view_count_count')
            ax2.barh(range(len(top_by_count)), top_by_count['view_count_count'],
                     color=plt.cm.plasma(np.linspace(0, 1, len(top_by_count))), alpha=0.8)
            ax2.set_yticks(range(len(top_by_count)))
            ax2.set_yticklabels([f'Topic {int(topic)}' for topic in top_by_count['primary_topic']])
            ax2.set_xlabel('Video numbers')
            ax2.set_title('Top 15 video numbers topics', fontsize=12, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            ax2.invert_yaxis()

            plt.suptitle('Topic rank', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('topic_detailed_ranking.png', dpi=600)
            plt.show()
            # =============================================

            topic_counts = self.df['primary_topic'].value_counts()
            topic_counts = topic_counts[topic_counts >= 10]
            topic_counts = topic_counts[topic_counts.index != -1]

            if len(topic_counts) > 0:
                plt.figure(figsize=(18, 12))
                top_topics = topic_counts.head(10)
                other_count = topic_counts.iloc[10:].sum()

                plot_values = list(top_topics.values) + [other_count]
                plot_labels = [f'Topic {i}\n({count})' for i, count in top_topics.items()] + \
                              [f'Other topic \n({other_count} videos)']

                colors = list(plt.cm.Set3(np.linspace(0, 1, 10))) + ['lightgray']

                wedges, texts, autotexts = plt.pie(plot_values,
                                                   labels=plot_labels,
                                                   autopct='%1.1f%%', startangle=90,
                                                   colors=colors, shadow=True)

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(10)

                plt.title('Topic distribution', fontsize=16, fontweight='bold', pad=20)
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig('topic_distribution_pie.png', dpi=600)
                plt.show()

        def performance_summary():
            print("6. Performance summary")
            if self.regression_results:
                plt.figure(figsize=(18, 12))

                models = list(self.regression_results.keys())
                r2_scores = [self.regression_results[model]['r2'] for model in models]
                model_names = [self.regression_results[model]['target_name'] for model in models]

                bars = plt.bar(model_names, r2_scores,
                               color=['steelblue', 'green', 'orange'],
                               alpha=0.8, edgecolor='black', linewidth=1)

                for bar, score in zip(bars, r2_scores):
                    plt.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.01,
                             f'{score:.4f}', ha='center', va='bottom',
                             fontsize=12, fontweight='bold')

                plt.title('Regression model performance Summary', fontsize=16, fontweight='bold')
                plt.xlabel('Target variable', fontsize=12)
                plt.ylabel('R² score', fontsize=12)
                plt.ylim(0, max(r2_scores) * 1.1)
                plt.grid(axis='y', alpha=0.3)

                plt.tight_layout()
                plt.savefig('model_performance_summary.png', dpi=600)
                plt.show()

        top_100()
        style_performance()
        feature_imp()
        title_impact()
        if not SAMPLE:
            topic_analysis()
        performance_summary()
        self.style_overview()

    def style_overview(self):
        long_df = self.df.explode("content_styles").dropna(subset=["content_styles"])

        style_stats = (
            long_df.groupby("content_styles")
                   .agg(videos      = ("display_id", "count"),
                        views_avg   = ("view_count", "mean"),
                        likes_avg   = ("like_count", "mean"),
                        comments_avg= ("num_comms", "mean"))
                   .assign(engage_avg = lambda d: (d["likes_avg"] + d["comments_avg"]) / d["views_avg"])
                   .sort_values("views_avg", ascending=False)
                   .round(2)
        )

        metrics_to_plot = {
            "views_avg": ("Average views", "style_avg_views.png", "Blues_d"),
            "likes_avg": ("Average likes", "style_avg_likes.png", "Reds_d"),
            "comments_avg": ("Average comments", "style_avg_comments.png", "Purples_d"),
        }

        fig, axes = plt.subplots(1, 3, figsize=(24, max(5, 0.4 * len(style_stats))))
        style_stats_reset = style_stats.reset_index()

        for ax, (m_col, (m_title, fname, palette)) in zip(axes, metrics_to_plot.items()):
            sorted_data = style_stats_reset.sort_values(by=m_col, ascending=False)

            sns.barplot(data=sorted_data,
                        y="content_styles", x=m_col,
                        palette=palette, edgecolor="black", ax=ax)
            ax.set_title(f"{m_title}", fontsize=15, fontweight="bold")
            ax.set_xlabel(m_title)
            ax.set_ylabel("Content style" if ax == axes[0] else "", fontsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
            ax.grid(axis='x', alpha=0.25)

        plt.tight_layout()
        plt.savefig("style_avg_combined.png", dpi=600)
        plt.show()

        plt.figure(figsize=(9, max(6, 0.35 * len(style_stats))))
        sns.heatmap(style_stats[["views_avg", "likes_avg", "comments_avg", "engage_avg"]],
                    annot=True, fmt=".1f", cmap="YlOrRd", linewidths=.5, cbar=True)
        plt.title("Content Style × Key Metrics Heatmap", fontsize=15, fontweight="bold", pad=16)
        plt.xlabel("Metric")
        plt.ylabel("Content style")
        plt.tight_layout()
        plt.savefig("style_metrics_heatmap.png", dpi=600)
        plt.show()

        self.style_stats = style_stats
        print(style_stats)


if __name__ == '__main__':
    analyser = Analyser()
    FLAG = os.path.exists("video_features_all.parquet")
    if FLAG:
        analyser.df = pd.read_parquet("video_features_all.parquet")
        analyser.top100_videos = analyser.df.nlargest(100, "view_count")
        print("skip to corelation analysis")
    if not FLAG:
        analyser.load_data()
        analyser.top_content()
        print("==============Content style analysis==============")
        analyser.content_style()
        if not SAMPLE:
            print("==============Topic model by bertopic==============")
            analyser.content_topic()
    analyser.select_features()
    print("==============Correlation analysis==============")
    analyser.correlation_analysis()
    print("==============Regression analysis==============")
    analyser.regression_analysis()
    print("==============Visualization==============")
    analyser.visualization()
