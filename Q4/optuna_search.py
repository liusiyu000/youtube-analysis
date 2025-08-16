import json
import lzma
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_FEAT   = Path("../Q2/video_features_all.parquet")
EMB_PATH    = Path("../Q1/embeddings_cpu.pt")
STUDY_DB    = Path("optuna_risk_study.db")   # SQLite
MODEL_OUT   = Path("best_xgb_model.xz")
METRIC_OUT  = Path("cv_metrics.json")
TRIALS      = 50
SEED        = 0

assert BASE_FEAT.exists(), "no video_features_all.parquet"
df = pd.read_parquet(BASE_FEAT)

EXAG_KW = [
    r"\bget\s*rich\s*quick\b", r"\beasy\s*money\b",
    r"\bguaranteed\s*(income|returns?)\b", r"\b100%\s*(profit|returns?)\b",
    r"\bdouble\s*your\s*money\b", r"\bpassive\s*income\s*overnight\b",
]
DISC_KW = [r"\bnot\s*financial\s*advice\b", r"\bdo\s*your\s*own\s*research\b"]
pat = lambda p: "(" + "|".join(p) + ")"
def nmatch(s, p): return s.str.lower().str.count(pat(p), flags=re.IGNORECASE).fillna(0)

df["exaggerate_cnt"]  = nmatch(df["title"],EXAG_KW)+nmatch(df["description"],EXAG_KW)
df["disclaimer_cnt"]  = nmatch(df["title"],DISC_KW)+nmatch(df["description"],DISC_KW)
df["emotion_strength"]=(df["title_sentiment_compound"].abs()*df["title_subjectivity"]).fillna(0)

rule1 = df["exaggerate_cnt"]>=1
rule2 = (df.get("title_has_dollar",0)==1)&(df.get("title_has_numbers",0)==1)&(df.get("title_has_caps",0)==1)
rule3 = (df["emotion_strength"]>0.55)&(df.get("title_has_exclamation",0)==1)
df["label"] = ((rule1|rule2|rule3)&(df["disclaimer_cnt"]<=1)).astype(int)

drop_cols = {"label", "risk_label", "risk_score", "exaggerate_cnt", "disclaimer_cnt", "emotion_strength", "title_has_dollar", "title_has_numbers", "title_has_caps", "title_has_exclamation", "title_sentiment_compound", "title_subjectivity", "title_sentiment_positive", "title_sentiment_negative", "title_sentiment_neutral", "description_sentiment_compound", "description_sentiment_positive", "description_sentiment_negative", "description_sentiment_neutral", "description_subjectivity"}
NUM_TYPES = {"int16","int32","int64","float16","float32","float64","bool"}

feat_cols = [
    c for c,t in df.dtypes.items()
    if str(t) in NUM_TYPES and c not in drop_cols
    and c not in {"video_id","channel_id","title","description"}
    and c not in {"predicted_topic", "primary_topic", "topic_confidence", "view_count_topic_avg",
                     "like_count_topic_avg", "view_vs_topic_avg", "like_vs_topic_avg"}

]
X_struct = df[feat_cols].fillna(0).values

if EMB_PATH.exists():
    emb = torch.load(EMB_PATH).astype("float32")
    assert emb.shape[0]==df.shape[0],"error"
else:
    emb = np.zeros((df.shape[0],384),dtype="float32")
X_full = np.hstack([X_struct,emb]).astype("float32")
y = df["label"].values

pos_idx = np.where(y==1)[0]
neg_idx = np.where(y==0)[0]
rng = np.random.default_rng(SEED)
neg_keep = rng.choice(neg_idx, size=min(len(pos_idx)*4,len(neg_idx)), replace=False)
keep = np.concatenate([pos_idx,neg_keep])
X = X_full[keep]; y_use=y[keep]

device = "cuda" if torch.cuda.is_available() else "cpu"
def objective(trial):
    params = {
        "n_estimators":   trial.suggest_int("n_estimators", 1000, 3000),
        "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":      trial.suggest_int("max_depth", 4, 10),
        "min_child_weight":trial.suggest_int("min_child_weight",1,10),
        "gamma":          trial.suggest_float("gamma", 0, 1),
        "subsample":      trial.suggest_float("subsample",0.6,1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree",0.6,1.0),
        "reg_lambda":     trial.suggest_float("reg_lambda",1e-2,10,log=True),
        "reg_alpha":      trial.suggest_float("reg_alpha",1e-3,1,log=True),
        "objective":"binary:logistic",
        "eval_metric":"auc",
        "tree_method":"auto",
        "random_state":SEED,
        "n_jobs":-1,
        "device":device,
        "early_stopping_rounds":50
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs, f1s = [], []
    for tr,va in skf.split(X,y_use):
        pos_w = (len(tr)-y_use[tr].sum())/y_use[tr].sum()
        params["scale_pos_weight"]=pos_w
        clf=XGBClassifier(**params)
        clf.fit(X[tr],y_use[tr],
                eval_set=[(X[va],y_use[va])],
                verbose=False)
        prob=clf.predict_proba(X[va])[:,1]
        aucs.append(roc_auc_score(y_use[va],prob))
        p,r,t=precision_recall_curve(y_use[va],prob)
        f1s.append(np.max(2*p*r/(p+r+1e-9)))
    trial.set_user_attr("f1_mean", float(np.mean(f1s)))
    return np.mean(aucs)

sampler=TPESampler(seed=SEED)
pruner =MedianPruner(n_warmup_steps=5)
study = optuna.create_study(direction="maximize",
                            sampler=sampler, pruner=pruner,
                            study_name="xgb_risk",
                            storage=f"sqlite:///{STUDY_DB}",
                            load_if_exists=True)
study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)

print("Best AUC:", study.best_value)
print("Best Params:\n", json.dumps(study.best_params,indent=2))

best=study.best_params
best.update({
    "objective":"binary:logistic",
    "eval_metric":"auc",
    "n_jobs":-1,
    "random_state":SEED,
    "scale_pos_weight":(len(y)-y.sum())/y.sum(),
})
clf_best=XGBClassifier(**best)
clf_best.fit(X_full,y)

with lzma.open(MODEL_OUT,"wb") as f:
    pickle.dump({"model":clf_best,"params":best,
                 "best_auc":study.best_value,
                 "best_f1":study.best_trial.user_attrs["f1_mean"],
                 "features":feat_cols},f)


with open(METRIC_OUT,"w") as fp:
    json.dump({"best_auc":study.best_value,
               "cv_f1":study.best_trial.user_attrs["f1_mean"],
               "params":best},fp,indent=2)

