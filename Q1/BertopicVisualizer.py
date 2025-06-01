import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class BERTopicVisualizer:
    def __init__(self, model_path="bertopic_model.pkl.lzma"):
        self.topic_model = joblib.load(model_path)

    def display_top_topics(self, n_topics=15):
        topic_info = self.topic_model.get_topic_info().head(n_topics)

        print("=" * 100)
        print(f"TOP {n_topics} TOPICS OVERVIEW")
        print("=" * 100)

        for index, row in topic_info.iterrows():
            topic_id = row['Topic']
            count = row['Count']

            if topic_id == -1:
                print(f"\nTopic #{topic_id} (Outliers) - Count: {count}")
            else:
                print(f"\nTopic #{topic_id} - Count: {count}")

            # Get topic words and scores
            topic_words = self.topic_model.get_topic(topic_id)

            if topic_words:
                print("Top 10 words:")
                for i, (word, score) in enumerate(topic_words[:10], 1):
                    print(f"  {i}. {word:<20} (score: {score:.4f})")
            print("-" * 50)

    def create_wordcloud_grid(self, n_topics=15, save_path="topic_wordclouds.png"):
        cols = 5
        rows = (n_topics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
        axes = axes.flatten()

        topic_info = self.topic_model.get_topic_info().head(n_topics)

        for idx, (_, row) in enumerate(topic_info.iterrows()):
            topic_id = row['Topic']

            topic_words = self.topic_model.get_topic(topic_id)

            if topic_words and topic_id != -1:
                word_freq = {word: score for word, score in topic_words[:30] if score > 0}

                if word_freq:
                    wordcloud = WordCloud(
                        width=400,
                        height=300,
                        background_color='white',
                        colormap='viridis',
                        relative_scaling=0.5,
                        min_font_size=10
                    ).generate_from_frequencies(word_freq)

                    axes[idx].imshow(wordcloud, interpolation='bilinear')
                    axes[idx].set_title(f'Topic {topic_id}', fontsize=12, fontweight='bold')
                    axes[idx].axis('off')
                else:
                    axes[idx].text(0.5, 0.5, f'Topic {topic_id}\n(No words)',
                                   ha='center', va='center', transform=axes[idx].transAxes)
                    axes[idx].axis('off')
            else:
                axes[idx].text(0.5, 0.5, f'Topic {topic_id}\n(Outliers)',
                               ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')

        for idx in range(n_topics, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        plt.show()

    def create_topic_heatmap(self, n_topics=15, n_words=10, save_path="topic_heatmap.png"):
        topic_info = self.topic_model.get_topic_info().head(n_topics)

        topics_words = []
        all_words = set()

        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:
                topic_words = self.topic_model.get_topic(topic_id)[:n_words]
                topic_dict = {word: score for word, score in topic_words}
                topics_words.append((topic_id, topic_dict))
                all_words.update(topic_dict.keys())

        all_words = sorted(list(all_words))
        matrix = []
        topic_labels = []

        for topic_id, word_dict in topics_words:
            row = [word_dict.get(word, 0) for word in all_words]
            matrix.append(row)
            topic_labels.append(f"Topic {topic_id}")

        matrix = np.array(matrix)

        plt.figure(figsize=(18, 12))
        sns.heatmap(matrix,
                    xticklabels=all_words,
                    yticklabels=topic_labels,
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Word Score'},
                    fmt='.3f')

        plt.xlabel('Words', fontsize=12)
        plt.ylabel('Topics', fontsize=12)
        plt.title('Topic-Word Score Heatmap', fontsize=14, fontweight='bold')
        plt.xticks(rotation=90, ha='right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        plt.show()

    def create_interactive_topic_viz(self, n_topics=15, save_path="interactive_topics.html"):
        topic_info = self.topic_model.get_topic_info().head(n_topics)

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Topic Distribution', 'Top Words per Topic'),
            row_heights=[0.3, 0.7],
            vertical_spacing=0.1
        )

        topic_info_filtered = topic_info[topic_info['Topic'] != -1]
        fig.add_trace(
            go.Bar(
                x=topic_info_filtered['Topic'].astype(str),
                y=topic_info_filtered['Count'],
                name='Document Count',
                marker_color='lightblue'
            ),
            row=1, col=1
        )

        for _, row in topic_info_filtered.iterrows():
            topic_id = row['Topic']
            topic_words = self.topic_model.get_topic(topic_id)[:10]

            if topic_words:
                words = [word for word, _ in topic_words]
                scores = [score for _, score in topic_words]

                fig.add_trace(
                    go.Bar(
                        x=scores,
                        y=words,
                        name=f'Topic {topic_id}',
                        orientation='h',
                        visible=True if topic_id == 0 else False
                    ),
                    row=2, col=1
                )

        fig.update_layout(
            title_text="BERTopic Analysis Results",
            showlegend=False,
            height=800,
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            label=f"Topic {row['Topic']}",
                            method="update",
                            args=[{"visible": [True] + [i == idx for i in range(len(topic_info_filtered) - 1)]}]
                        )
                        for idx, (_, row) in enumerate(topic_info_filtered.iterrows())
                    ],
                    direction="down",
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=0.5,
                    yanchor="top"
                )
            ]
        )

        fig.update_xaxes(title_text="Topic ID", row=1, col=1)
        fig.update_yaxes(title_text="Document Count", row=1, col=1)
        fig.update_xaxes(title_text="Score", row=2, col=1)
        fig.update_yaxes(title_text="Words", row=2, col=1)

        fig.write_html(save_path)
        print(f"Interactive visualization saved to {save_path}")

    def save_topic_details_to_excel(self, n_topics=15, save_path="topic_details.xlsx"):
        with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
            topic_info = self.topic_model.get_topic_info().head(n_topics)
            topic_info.to_excel(writer, sheet_name='Overview', index=False)

            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                if topic_id != -1:
                    topic_words = self.topic_model.get_topic(topic_id)[:20]

                    if topic_words:
                        df_words = pd.DataFrame(topic_words, columns=['Word', 'Score'])
                        df_words['Rank'] = range(1, len(df_words) + 1)
                        df_words = df_words[['Rank', 'Word', 'Score']]

                        sheet_name = f'Topic_{topic_id}'
                        df_words.to_excel(writer, sheet_name=sheet_name, index=False)

                        worksheet = writer.sheets[sheet_name]
                        worksheet.set_column('A:A', 10)
                        worksheet.set_column('B:B', 30)
                        worksheet.set_column('C:C', 15)

        print(f"Topic details saved to {save_path}")


if __name__ == '__main__':
    viz = BERTopicVisualizer()

    viz.display_top_topics(n_topics=15)

    viz.create_wordcloud_grid(n_topics=15)

    viz.create_topic_heatmap(n_topics=15, n_words=10)

    viz.create_interactive_topic_viz(n_topics=15)

    viz.save_topic_details_to_excel(n_topics=15)