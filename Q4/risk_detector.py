import lzma
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


warnings.filterwarnings("ignore")

BASE_FEAT   = Path("../Q2/video_features_all.parquet")
EMB_PATH    = Path("../Q1/embeddings_cpu.pt")
OUT_PARQUET = Path("./video_features_with_risk_new1.parquet")
MODEL_PATH  = Path("./risk_classifier_pack_new1.xz")

assert BASE_FEAT.exists(), "Please run top_video_analyser.py first to generate video_features_all.parquet"
df = pd.read_parquet(BASE_FEAT)
print("Loading feature table:", df.shape)


EXAG_KW = [
    r"\bget\s*rich\s*quick\b", r"\beasy\s*money\b",
    r"\bguaranteed\s*(income|returns?)\b", r"\b100%\s*(profit|returns?)\b",
    r"\bdouble\s*your\s*money\b", r"\bpassive\s*income\s*overnight\b",
]
DISC_KW = [
    r"\bnot\s*financial\s*advice\b", r"\bdo\s*your\s*own\s*research\b"
]
def nmatch(s, patterns):
    pat = "(" + "|".join(patterns) + ")"
    return s.str.lower().str.count(pat, flags=re.IGNORECASE).fillna(0)

df["exaggerate_cnt"]  = nmatch(df["title"], EXAG_KW) + nmatch(df["description"], EXAG_KW)
df["disclaimer_cnt"]  = nmatch(df["title"], DISC_KW) + nmatch(df["description"], DISC_KW)
df["emotion_strength"] = (df["title_sentiment_compound"].abs()
                          * df["title_subjectivity"]).fillna(0)

rule1 = df["exaggerate_cnt"] >= 1
rule2 = (df.get("title_has_dollar",0)==1) & (df.get("title_has_numbers",0)==1) & (df.get("title_has_caps",0)==1)
rule3 = (df["emotion_strength"]>0.55) & (df.get("title_has_exclamation",0)==1)
df["label"] = ((rule1 | rule2 | rule3) & (df["disclaimer_cnt"] <= 1)).astype(int)
print("Weak label distribution:\n", df["label"].value_counts())


drop_cols = {"label", "risk_label", "risk_score", "exaggerate_cnt", "disclaimer_cnt", "emotion_strength", "title_has_dollar", "title_has_numbers", "title_has_caps", "title_has_exclamation", "title_sentiment_compound", "title_subjectivity", "title_sentiment_positive", "title_sentiment_negative", "title_sentiment_neutral", "description_sentiment_compound", "description_sentiment_positive", "description_sentiment_negative", "description_sentiment_neutral", "description_subjectivity"}

NUMERIC_TYPES = {"int16","int32","int64","float16","float32","float64","bool"}


cand_cols = [
    c for c,typ in df.dtypes.items()
    if (str(typ) in NUMERIC_TYPES) and (c not in drop_cols)
    if (str(typ) in NUMERIC_TYPES)
    and c not in {"video_id","channel_id","title","description"}
    and c not in {"predicted_topic", "primary_topic", "topic_confidence", "view_count_topic_avg", "like_count_topic_avg", "view_vs_topic_avg", "like_vs_topic_avg"}
]

X_struct = df[cand_cols].fillna(0).values
print("Number of structural features:", len(cand_cols))

if EMB_PATH.exists():
    import torch
    emb = torch.load(EMB_PATH)      # (N, 384)
    assert emb.shape[0] == df.shape[0], "embeddings_cpu.pt does not match the number of rows in the DataFrame"
    X_text = emb.astype("float32")
else:
    exit(1)

X_full = np.hstack([X_struct, X_text]).astype("float32")
y = df["label"].values

if MODEL_PATH.exists():
    print(f"\n✓ Found the saved model file: {MODEL_PATH}")
    print("→ Load the model, skipping the training step...")

    with lzma.open(MODEL_PATH, "rb") as f:
        model_pack = pickle.load(f)

    fold_models = model_pack["models"]
    saved_features = model_pack["features_struct"]
    global_thr = model_pack["global_threshold"]

    if set(saved_features) != set(cand_cols):
        print("Warning: The current feature columns are not exactly the same as those used during model training")
        print(f"  - Number of model features: {len(saved_features)}")
        print(f"  - Current number of features: {len(cand_cols)}")
        missing = set(saved_features) - set(cand_cols)
        extra = set(cand_cols) - set(saved_features)
        if missing:
            print(f"  - Missing features: {missing}")
        if extra:
            print(f"  - Redundant features: {extra}")

        try:
            X_struct_aligned = df[saved_features].fillna(0).values
            X_full = np.hstack([X_struct_aligned, X_text]).astype("float32")
            print("  → Features have been rearranged as required by the model")
        except KeyError as e:
            print(f"  ✗ Unable to align features, required column missing: {e}")
            raise

    print(f"→ load {len(fold_models)} model")
    print(f"→ Global Threshold: {global_thr:.3f}")

else:
    print("\n→ No saved model found, start training...")

    pos_idx = np.where(y==1)[0]
    neg_idx = np.where(y==0)[0]
    rng = np.random.default_rng(0)
    neg_keep = rng.choice(neg_idx, size=min(len(pos_idx)*4, len(neg_idx)), replace=False)
    keep_idx = np.concatenate([pos_idx, neg_keep])
    X = X_full[keep_idx]; y_use = y[keep_idx]
    print("Shape after sampling:", X.shape, "  +:", y_use.sum(), "  -:", len(y_use)-y_use.sum())

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    fold_models, fold_aucs, fold_thresh = [], [], []

    for i, (tr, va) in enumerate(skf.split(X, y_use), 1):
        pos_wt = (len(tr)-y_use[tr].sum())/y_use[tr].sum()
        clf = XGBClassifier(
            n_estimators=3000,
            max_depth=10,
            learning_rate=0.042755,
            subsample=0.9554, colsample_bytree=0.962,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=pos_wt,
            n_jobs=-1, random_state=i*7,
            min_child_weight=10,
            gamma=0.1881,
            reg_lambda=5.732,
            reg_alpha=0.00194,
            device= "cuda" if torch.cuda.is_available() else "cpu",
            early_stopping_rounds=50,
        )
        clf.fit(X[tr], y_use[tr],
                eval_set=[(X[va],y_use[va])],
                verbose=50)

        prob_va = clf.predict_proba(X[va])[:,1]
        auc = roc_auc_score(y_use[va], prob_va)
        p, r, t = precision_recall_curve(y_use[va], prob_va)
        f1 = 2*p*r/(p+r+1e-5)
        best_idx = np.argmax(f1)
        fold_aucs.append(auc)
        fold_thresh.append(t[best_idx])
        fold_models.append(clf)
        print(f"Fold{i}: AUC={auc:.4f}  bestF1={f1[best_idx]:.4f} @thr={t[best_idx]:.3f}")

    global_thr = float(np.median(fold_thresh))
    print(f"→ 50% fold AUC mean {np.mean(fold_aucs):.4f}   Using global threshold {global_thr:.3f}")

    with lzma.open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "models": fold_models,
            "features_struct": cand_cols,
            "global_threshold": global_thr
        }, f)
    print("✓ Model package saved →", MODEL_PATH)

print("\n→ Start full prediction...")
prob_sum = np.zeros(df.shape[0], dtype="float32")
for clf in fold_models:
    prob_sum += clf.predict_proba(X_full)[:,1]
prob_avg = prob_sum / len(fold_models)

df["risk_score"] = prob_avg
df["risk_label"] = (df["risk_score"] >= global_thr).astype(int)

print("\n→ Classification Report (risk_label vs. weak label):")
print(classification_report(df["label"], df["risk_label"],
      target_names=["money-not-related", "money-related"]))

print(f"→ Risk label distribution:")
print(df["risk_label"].value_counts())
print(f"→ Risk score statistics: mean={df['risk_score'].mean():.3f}, "
      f"std={df['risk_score'].std():.3f}, "
      f"max={df['risk_score'].max():.3f}")

print("\n→ risk video sample(risk_label=1):")
risk_videos = df[df["risk_label"] == 1].sort_values("risk_score", ascending=False).head(5)
for idx, row in risk_videos.iterrows():
    print(f"Title: {row['title']}")
    print(f"Risk score: {row['risk_score']:.3f}")
    print(f"Rules triggering: exaggerate={row['exaggerate_cnt']}, emotion={row['emotion_strength']:.2f}")
    print(f"Video description: {row['description'][:200]}..." if len(str(row['description'])) > 200 else f"description: {row['description']}")
    print("-" * 80)

df.to_parquet(OUT_PARQUET, compression="snappy")
