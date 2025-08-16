import os
import re
import string
import jsonlines
import pandas as pd
import matplotlib.pyplot as plt

REVENUE_PATTERNS = {
    "affiliate": ["affiliate", "referral", "commission", "promo code", "discount code", "coupon", "link in bio",
                  "link below", "use my link", "special offer"],
    "sponsored": ["sponsored", "partnership", "collaboration", "brand ambassador", "paid promotion", "thanks to",
                  "supported by", "brought to you by"],
    "course": ["course", "training", "masterclass", "bootcamp", "workshop", "coaching", "consultation", "mentoring",
               "ebook"],
    "product": ["merch", "merchandise", "shop", "store", "app", "software", "subscription", "membership", "premium"],
    "donation": ["patreon", "paypal", "donate", "support", "tip jar", "ko-fi"],
}

VIDEO_PATH = "../Q1/money_related_content.jsonl"  # Video metadata with title, description, upload_date


def load_and_preprocess(video_path):
    with jsonlines.open(video_path, "r") as reader:
        data = [obj for obj in reader]
    df = pd.DataFrame(data)

    df['upload_date'] = pd.to_datetime(df['upload_date'].str.split(' ').str[0])
    df['quarter'] = df['upload_date'].dt.to_period('Q').astype(str)

    df = df[(df['upload_date'].dt.year >= 2015) & (df['upload_date'].dt.year <= 2019)]

    return df


def detect_external_strategies(df):

    def detect_keywords(text, keywords):
        return {kw: text.str.contains(kw, case=False, na=False).astype(int) for kw in keywords}

    df['title_desc'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    strategy_dfs = []
    for rev_type, keywords in REVENUE_PATTERNS.items():
        sub_df = pd.DataFrame(detect_keywords(df['title_desc'], keywords))
        sub_df.columns = [f"has_{rev_type}_{kw.replace(' ', '_')}" for kw in keywords]
        sub_df[f"has_{rev_type}"] = (sub_df.sum(axis=1) > 0).astype(int)
        strategy_dfs.append(sub_df)

    strategies = pd.concat(strategy_dfs, axis=1)
    return pd.concat([df, strategies], axis=1)


def temporal_analysis(df):
    """Compute quarterly mean usage rates for main strategy categories only."""
    strategy_cols = ['has_affiliate', 'has_sponsored', 'has_course', 'has_product', 'has_donation']
    strategy_cols = [col for col in strategy_cols if col in df.columns]
    grouped = df.groupby('quarter')[strategy_cols].mean().reset_index()
    return grouped


def visualize_temporal(grouped, output_file):
    """Generate line plot for main strategy categories only."""
    plt.figure(figsize=(12, 8))

    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', '^', 's', 'd', 'x']
    colors = plt.cm.tab10.colors

    for i, strategy in enumerate(grouped.columns[1:]):
        plt.plot(grouped['quarter'], grouped[strategy],
                 label=strategy.replace('has_', '').capitalize(),
                 color=colors[i % len(colors)],
                 linestyle=line_styles[i % len(line_styles)],
                 marker=markers[i % len(markers)],
                 linewidth=2, alpha=0.8)

    plt.title("Evolution of Money-Making Strategies Over Time")
    plt.xlabel("Quarter")
    plt.ylabel("Proportion (Mean Usage Rate)")
    plt.xticks(rotation=45)
    plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.show()


def temporal_analysis_keywords(df):
    sub_cols = {rt: [col for col in df.columns if col.startswith(f"has_{rt}_")] for rt in REVENUE_PATTERNS.keys()}
    grouped_subs = {}
    for rt, cols in sub_cols.items():
        if cols:
            grouped_subs[rt] = df.groupby('quarter')[cols].mean().reset_index()
    return grouped_subs


def visualize_strategy_keywords(grouped_subs, output_file):
    n_categories = len(grouped_subs)
    fig, axes = plt.subplots(n_categories, 1, figsize=(12, 4 * n_categories), sharex=True)
    if n_categories == 1: axes = [axes]

    colors = plt.cm.tab10.colors
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', '^', 's', 'd']

    for i, (category, sub_grouped) in enumerate(grouped_subs.items()):
        ax = axes[i]
        for j, sub_strategy in enumerate(sub_grouped.columns[1:]):  # Skip 'quarter'
            clean_label = sub_strategy.replace(f"has_{category}_", "").replace('_', ' ').capitalize()
            ax.plot(sub_grouped['quarter'], sub_grouped[sub_strategy],
                    label=clean_label,
                    color=colors[j % len(colors)],
                    linestyle=line_styles[j % len(line_styles)],
                    marker=markers[j % len(markers)],
                    linewidth=2, alpha=0.8)

        ax.set_title(f"Evolution of Sub-Keywords in {category.capitalize()} Strategy Over Time")
        ax.set_ylabel("Proportion (Mean Usage Rate)")
        ax.legend(title="Sub-Keyword", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    axes[-1].set_xlabel("Quarter")
    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.show()


def main():
    df = load_and_preprocess(VIDEO_PATH)
    df = detect_external_strategies(df)

    grouped = temporal_analysis(df)
    visualize_temporal(grouped, "temporal_strategies_line.png")

    grouped_subs = temporal_analysis_keywords(df)
    visualize_strategy_keywords(grouped_subs, "temporal_strategy_keywords_subplots.png")

if __name__ == "__main__":
    main()