import json
import lzma
import pickle

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

total_lines = 534478
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    with lzma.open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def stream_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def process_and_predict(model, text):
    text_embedding = embedder.encode([text])
    prob = model.predict_proba(text_embedding)[:, 1]
    return prob


def save_money_related_entries(input_file, output_file, model, batch_size=256):
    batch_texts = []
    batch_rows = []

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for row in tqdm(stream_jsonl(input_file), total=total_lines, desc="Processing records"):
            text = "\n".join([
                str(row.get("title", "")).strip(),
                str(row.get("tags", "")).strip(),
                str(row.get("description", "")).strip()
            ]).strip()

            if not text:
                continue

            batch_texts.append(text)
            batch_rows.append(row)

            if len(batch_texts) == batch_size:
                embeddings = embedder.encode(batch_texts, batch_size=batch_size, device="cuda" if torch.cuda.is_available() else "cpu", show_progress_bar=False)
                probs = model.predict_proba(embeddings)[:, 1]

                for prob, r in zip(probs, batch_rows):
                    if prob >= 0.5:
                        r['predicted_topic'] = 1
                        r['prediction_prob'] = float(prob)
                        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

                batch_texts = []
                batch_rows = []

        if batch_texts:
            embeddings = embedder.encode(batch_texts, batch_size=batch_size, device="cuda" if torch.cuda.is_available() else "cpu", show_progress_bar=False)
            probs = model.predict_proba(embeddings)[:, 1]
            for prob, r in zip(probs, batch_rows):
                if prob >= 0.5:
                    r['predicted_topic'] = 1
                    r['prediction_prob'] = float(prob)
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    model_path = 'hustle_classifier_expend.xz'
    input_file = 'F:\dissertationData\yt_metadata_en_filtered_big.jsonl'
    output_file = 'money_related_content.jsonl'

    model = load_model(model_path)

    save_money_related_entries(input_file, output_file, model)
