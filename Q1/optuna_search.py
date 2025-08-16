import argparse
import json
import os
from pathlib import Path

import nltk
import numpy as np
import optuna
import torch
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

nltk.download('wordnet')
os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/temp'


MONEY_KEYWORDS = [
    "money","income","earning","profit","revenue",
    "dropshipping","ecommerce","marketing alliance", "freelance","investing",
    "cryptocurrency","stock","trading", "$", "dollar", "hustle", "wealth",
    "shopify","sponsorship", "investment",
]
MONEY_KW_LOWER = [k.lower() for k in MONEY_KEYWORDS]
def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

expanded_keywords = []
for keyword in MONEY_KEYWORDS:
    expanded_keywords.extend(get_synonyms(keyword))
expanded_keywords = set(expanded_keywords)

def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

expanded_keywords = []
for keyword in MONEY_KEYWORDS:
    expanded_keywords.extend(get_synonyms(keyword))
expanded_keywords = set(expanded_keywords)

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def soft_label(text):
    t = text.lower()
    return int(any(kw in t for kw in expanded_keywords))


def main(sample_path, model_out, report_out, seed):
    texts = []
    labels = []

    for row in read_jsonl(sample_path):
        txt = row.get("__text__", "").strip()
        if not txt:
            continue
        texts.append(txt)
        labels.append(soft_label(txt))

    print(f"Loaded {len(texts):,} samples. Hustle related={sum(labels):,}")

    if os.path.exists("./embeddings.pt"):
        # start_time = time.time()
        embeddings = torch.load("./embeddings.pt")
        # end_time = time.time()
        # print(f"torch.load() Time: {end_time-start_time:.4f} seconds") # 33.75s

        # start_time = time.time()
        # with lzma.open("./embeddings.pt.xz", 'rb') as f:
        #     embeddings = pickle.load(f)
        # end_time = time.time()
        # print(f"lzma.open() + pickle.load() Time: {end_time-start_time:.4f} seconds") # 158.76s
    else:
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
        embeddings = embedder.encode(texts, batch_size=128, show_progress_bar=True)
        torch.save(embeddings, "./embeddings.pt")

    y = np.array(labels)
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, y, test_size=0.2, random_state=seed, stratify=y
    )

    def optuna_search(trial):
        scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()
        param = {
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
            'max_depth': trial.suggest_int("max_depth", 4, 11),
            'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
            'gamma': trial.suggest_float("gamma", 0, 1.0),
            'subsample': trial.suggest_float("subsample", 0.6, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
            'scale_pos_weight': scale_pos,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'auto',
            'n_jobs': -1,
            'random_state': seed,
            'device': "cuda" if torch.cuda.is_available() else "cpu",
            "early_stopping_rounds": 50,
        }

        model = XGBClassifier(**param)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        return auc

    study = optuna.create_study(direction="maximize")  # maximize AUC
    study.optimize(optuna_search, n_trials=50)
    print("Best params: ", study.best_params)
    print("Best AUC :", study.best_value)

if __name__ == "__main__":
    p = argparse.ArgumentParser("train money/non-money classifier")
    p.add_argument("--sample_jsonl", type=Path, default="F:\dissertationData\yt_metadata_en_sample.jsonl", help="jsonl from create sample")
    p.add_argument("--model-out",   type=Path, default=Path("hustle_classifier.xz"))
    p.add_argument("--report-out",  type=Path, default=Path("valset_report_xgb.txt"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    main(args.sample_jsonl, args.model_out, args.report_out, args.seed)
