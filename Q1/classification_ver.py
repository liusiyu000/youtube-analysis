import json
import lzma
import pickle
import numpy as np
import pandas as pd
import torch.cuda
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report

INPUT_JSONL = "yt_labeled_by_human_and_gpt.jsonl"
MODEL_PATH = "./hustle_classifier_expend.xz"
OUTPUT_CSV = "money_predictions.csv"
THRESHOLD = 0.5
BATCH_SIZE = 128


def prepare_text(row: dict) -> str:
    title = str(row.get("title", "")).strip()
    tags = str(row.get("tags", "")).strip()
    desc = str(row.get("description", "")).strip()

    return "\n".join([title, tags, desc]).strip()


def load_money_model(path):
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def main():
    # Load data
    data = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples")

    # Load encoder and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    model = load_money_model(MODEL_PATH)

    # Text to embeddings
    texts = [prepare_text(r) for r in data]
    embeddings = embedder.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
    probs = model.predict_proba(embeddings)[:, 1]

    preds = (probs >= THRESHOLD).astype(int)

    # Save predictions
    df_raw = pd.DataFrame(data)
    out = pd.DataFrame()
    for c in ["display_id", "video_id", "id", "title"]:
        if c in df_raw.columns:
            out[c] = df_raw[c]
    out["money_prob"] = probs
    out["money_pred"] = preds
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved predictions to: {OUTPUT_CSV}")

    y_true = df_raw["money_label"].values  # Ground truth labels from annotated data

    print("\n===== Classification Report =====")
    print(classification_report(y_true, preds,
                                target_names=["Not Money", "Money"],
                                digits=4))


if __name__ == "__main__":
    main()