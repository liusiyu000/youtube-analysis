import argparse, json, random, sys
from pathlib import Path

TARGET_BYTES = 2 * 1024**3
ENC = "utf-8"
# TOTAL_LINES = 72,924,794

def build_sample(src, dst, prob, seed):
    rng = random.Random(seed)
    written = 0
    kept = 0

    with src.open("r", encoding=ENC) as fin, dst.open("w", encoding=ENC) as fout:
        for ln, raw in enumerate(fin, 1):
            if rng.random() > prob:
                continue

            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue

            row["__text__"] = "\n".join([
                str(row.get("title", "")).strip(),
                str(row.get("tags", "")).strip(),
                str(row.get("description", "")).strip()
            ]).strip()

            out = json.dumps(row, ensure_ascii=False)
            fout.write(out + "\n")

            written += len(raw.encode(ENC))
            kept += 1
            if written >= TARGET_BYTES:
                break

            if ln % 10_000 == 0:
                mb = written / 1024**2
                print(f"...read {ln:,} lines, sample {mb:,.1f} MB", file=sys.stderr)

    mb = written / 1024**2
    print(f"finished: {kept:,} lines → {mb:,.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser("2 GB sampler")
    p.add_argument("--input",  type=Path, default= "F:\dissertationData\yt_metadata_en.jsonl", help="Input path")
    p.add_argument("--output", type=Path, default="F:\dissertationData\yt_metadata_en_sample.jsonl", help="Output path")
    p.add_argument("--prob", type=float, default=2/95, help="sampling probability per line")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    args = p.parse_args()

    build_sample(args.input, args.output, args.prob, args.seed)
