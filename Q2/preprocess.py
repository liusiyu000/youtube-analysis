import jsonlines
import numpy as np
import pandas as pd

sample = False

# save joined table
class Preprocess:
    def __init__(self):
        self.join_df = None
        self.comment_nums_df = None
        self.timeseries_df = None
        self.channels_df = None
        self.videos_df = None
        if sample:
            self.channel_path = r"F:\dissertationData\top_1000_df_channels_en.tsv"
            self.metadata_path = r"F:\dissertationData\top_1000_yt_metadata_en_filtered_big.jsonl"
            self.comment_nums_path = r"F:\dissertationData\top_1000_num_comments.tsv"
            self.timeseries_path = r"F:\dissertationData\top_1000_df_timeseries_en.tsv"
            self.save_path = "top_1000_popular_analysis.parquet"
        else:
            self.channel_path = r"F:\dissertationData\df_channels_en.tsv"
            self.metadata_path = r"../Q1/money_related_content.jsonl"
            self.comment_nums_path = r"F:\dissertationData\num_comments.tsv"
            self.timeseries_path = r"F:\dissertationData\df_timeseries_en.tsv"
            self.save_path = "popular_analysis.parquet"


    def load_data(self):
        self.channels_df = pd.read_csv(self.channel_path, sep='\t')

        self.timeseries_df = pd.read_csv(self.timeseries_path, sep='\t')

        with jsonlines.open(self.metadata_path, "r") as reader:
            videos_list = [obj for obj in reader]
        self.videos_df = pd.DataFrame(videos_list)

        self.comment_nums_df = pd.read_csv(self.comment_nums_path, sep='\t')

        print(f"channel shape: {self.channels_df.shape}")
        print(f"timeseries shape: {self.timeseries_df.shape}")
        print(f"video metadata shape: {self.videos_df.shape}")
        print(f"comment nums shaps: {self.comment_nums_df.shape}")

    def join_table(self):
        self.videos_df['title'] = self.videos_df['title'].fillna('')
        self.videos_df['description'] = self.videos_df['description'].fillna('')
        self.videos_df['tags'] = self.videos_df['tags'].fillna('')

        numeric_cols = ['view_count', 'like_count', 'dislike_count', 'duration']
        for col in numeric_cols:
            self.videos_df[col] = pd.to_numeric(self.videos_df[col], errors='coerce').fillna(0)

        # metadata + comments_nums
        self.join_df = self.videos_df.copy()
        self.comment_nums_df['num_comms'] = self.comment_nums_df['num_comms'].fillna(0)
        self.join_df = self.join_df.merge(
            self.comment_nums_df,
            left_on='display_id',
            right_on='display_id',
            how='left'
        )

        # + channel data
        channel_basic = self.channels_df[['channel', 'name_cc', 'category_cc', 'subscribers_cc', 'videos_cc']].copy()
        self.join_df = self.join_df.merge(
            channel_basic,
            left_on='channel_id',
            right_on='channel',
            how='left'
        )

        # + timeseries data
        latest_timeseries = self.timeseries_df.drop_duplicates().sort_values('datetime').groupby('channel').tail(
            1)  # latest data
        timeseries_cols = ['channel', 'views', 'activity']
        latest_timeseries = latest_timeseries[timeseries_cols].copy()
        latest_timeseries.columns = ['channel', 'channel_total_views', 'channel_activity']
        self.join_df = self.join_df.merge(
            latest_timeseries,
            left_on='channel_id',
            right_on='channel',
            how='left'
        )

    def calc_metrix(self):
        self.join_df['engagement_rate'] = (self.join_df['like_count'] + self.join_df['num_comms']) / self.join_df[
            'view_count'].replace(0, 1)
        self.join_df['like_rate'] = self.join_df['like_count'] / self.join_df['view_count'].replace(0, 1)
        self.join_df['channel_avg_views_per_video'] = self.join_df['channel_total_views'] / self.join_df[
            'videos_cc'].replace(0, 1)
        self.join_df['video_popularity_vs_subs'] = self.join_df['view_count'] / self.join_df['subscribers_cc'].replace(
            0, 1)
        self.join_df = self.join_df.replace([np.inf, -np.inf], np.nan)

    def output(self):
        string_columns = ['category_cc', 'name_cc', 'title', 'description', 'tags', 'categories']
        for col in string_columns:
            self.join_df[col] = self.join_df[col].fillna('').astype(str)

        numeric_columns = [
            'view_count', 'like_count', 'dislike_count', 'duration', 'num_comms',
            'subscribers_cc', 'videos_cc', 'channel_total_views', 'channel_activity', 'engagement_rate', 'like_rate',
            'channel_avg_views_per_video', 'video_popularity_vs_subs'
        ]
        for col in numeric_columns:
            self.join_df[col] = pd.to_numeric(self.join_df[col], errors='coerce').fillna(0)

        cols_to_drop = ['channel_x', 'channel_y']
        self.join_df = self.join_df.drop(columns=cols_to_drop)

        self.join_df.to_parquet(self.save_path, compression='snappy', index=False)
        print(f"join table shape: {self.join_df.shape}")
        print(f"join table attributes: {list(self.join_df.columns)}")
        print(f"join table first 10 lines: {self.join_df.head(10)}")


if __name__ == '__main__':
    preprocess = Preprocess()
    preprocess.load_data()
    preprocess.join_table()
    preprocess.calc_metrix()
    preprocess.output()
