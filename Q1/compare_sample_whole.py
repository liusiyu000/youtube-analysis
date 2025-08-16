import json
import os
import numpy as np
from collections import Counter
from scipy.stats import ks_2samp
import random
import sys

# Define fields to analyze based on provided data structure
NUMERIC_FIELDS = ['view_count', 'like_count', 'dislike_count', 'duration']
CATEGORICAL_FIELDS = ['categories']

class OnlineStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.values = []  # For small files or reservoir sampling

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)
        self.values.append(x)

    def variance(self):
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)

    def std(self):
        return np.sqrt(self.variance())

    def median(self):
        if not self.values:
            return 0.0
        return np.median(self.values)

# Streaming counter for categorical variables
class OnlineCounter:
    def __init__(self):
        self.counter = Counter()

    def update(self, cat):
        if cat is not None:
            self.counter[cat] += 1

    def top_n(self, n=5):
        return self.counter.most_common(n)

# Reservoir sampling for large files
def reservoir_sample(stream, field, k=10000):
    reservoir = []
    for i, item in enumerate(stream):
        val = item.get(field)
        if isinstance(val, (int, float)):
            if len(reservoir) < k:
                reservoir.append(val)
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = val
    return reservoir

# Process a JSONL file (small or large)
def process_file(file_path, is_large=False, reservoir_size=10000):
    stats = {field: OnlineStats() for field in NUMERIC_FIELDS}
    cats = {field: OnlineCounter() for field in CATEGORICAL_FIELDS}
    reservoirs = {field: [] for field in NUMERIC_FIELDS}
    indices = {field: 0 for field in NUMERIC_FIELDS}
    line_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            if line_count % 100000 == 0:
                print(f"Processed {line_count} lines from {file_path}", file=sys.stderr)
            try:
                data = json.loads(line)
                for field in NUMERIC_FIELDS:
                    val = data.get(field)
                    if isinstance(val, (int, float)):
                        stats[field].update(val)
                        res = reservoirs[field]
                        i = indices[field]
                        if len(res) < reservoir_size:
                            res.append(val)
                        else:
                            j = random.randint(0, i)
                            if j < reservoir_size:
                                res[j] = val
                        indices[field] += 1
                for field in CATEGORICAL_FIELDS:
                    cat = data.get(field)
                    cats[field].update(cat)
            except json.JSONDecodeError:
                continue

    for field in NUMERIC_FIELDS:
        stats[field].values = reservoirs[field]

    # Compile results
    results = {}
    for field in NUMERIC_FIELDS:
        s = stats[field]
        results[field] = {
            'count': s.n,
            'mean': s.mean,
            'std': s.std(),
            'min': s.min_val,
            'max': s.max_val,
            'median': s.median() if s.values else 0.0
        }
    for field in CATEGORICAL_FIELDS:
        results[field] = {
            'top_5': cats[field].top_n(5),
            'unique_count': len(cats[field].counter)
        }
    results['total_lines'] = line_count
    return results, reservoirs

# Compare datasets and perform KS test
def compare_datasets(sample_path, full_path):
    print("Processing sample file...")
    sample_results, sample_samples = process_file(sample_path, is_large=False)

    print("Processing full file...")
    full_results, full_samples = process_file(full_path, is_large=True)

    # Print comparative statistics
    print("\n=== Descriptive Statistics Comparison ===")
    for field in NUMERIC_FIELDS:
        print(f"\nField: {field}")
        print("Sample Dataset:")
        print(f"  Count: {sample_results[field]['count']}")
        print(f"  Mean: {sample_results[field]['mean']:.2f}")
        print(f"  Std: {sample_results[field]['std']:.2f}")
        print(f"  Min: {sample_results[field]['min']:.2f}")
        print(f"  Max: {sample_results[field]['max']:.2f}")
        print(f"  Median: {sample_results[field]['median']:.2f}")
        print("Full Dataset:")
        print(f"  Count: {full_results[field]['count']}")
        print(f"  Mean: {full_results[field]['mean']:.2f}")
        print(f"  Std: {full_results[field]['std']:.2f}")
        print(f"  Min: {full_results[field]['min']:.2f}")
        print(f"  Max: {full_results[field]['max']:.2f}")
        print(f"  Median: {full_results[field]['median']:.2f}")

    for field in CATEGORICAL_FIELDS:
        print(f"\nField: {field}")
        print("Sample Dataset:")
        print(f"  Top 5 categories: {sample_results[field]['top_5']}")
        print(f"  Unique categories: {sample_results[field]['unique_count']}")
        print("Full Dataset:")
        print(f"  Top 5 categories: {full_results[field]['top_5']}")
        print(f"  Unique categories: {full_results[field]['unique_count']}")

    print(f"\nTotal entries: Sample={sample_results['total_lines']}, Full={full_results['total_lines']}")

    # KS test to prove sampling reasonableness
    print("\n=== KS Test for Distribution Similarity ===")
    print("(p-value > 0.05 suggests distributions are similar)")
    for field in NUMERIC_FIELDS:
        if sample_samples[field] and full_samples[field]:
            ks_stat, p_value = ks_2samp(sample_samples[field], full_samples[field])
            print(f"{field}:")
            print(f"  KS statistic: {ks_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value > 0.05:
                print(f"  -> Distributions are similar (fail to reject null hypothesis)")
            else:
                print(f"  -> Distributions may differ; consider stratified sampling")
        else:
            print(f"{field}: Insufficient data for KS test")

if __name__ == "__main__":

    sample_path = 'F:\dissertationData\yt_metadata_en_sample.jsonl'
    full_path = 'F:\dissertationData\yt_metadata_en.jsonl'
    if os.path.exists(sample_path) and os.path.exists(full_path):
        compare_datasets(sample_path, full_path)
    else:
        print("Files not found. Please ensure paths are correct.")