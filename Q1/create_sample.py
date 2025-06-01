import jsonlines, random, re, pathlib
import multiprocessing as mp
from functools import partial

input_path      = pathlib.Path("F:/dissertationData/yt_metadata_en.jsonl")
output_all      = pathlib.Path("F:/dissertationData/yt_metadata_en_money.jsonl")
output_sample50 = pathlib.Path("F:/dissertationData/yt_metadata_en_50k.jsonl")
sample_size     = 50000


intent_keywords = [
    r"make money", r"earn money", r"income", r"invest", r"investment", r"trading", r"profit", r"forex",
    r"side hustle", r"financial freedom", r"quit your job", r"stock", r"stocks", r"revenue",
    r"ways to earn", r"how to make", r"btc", r"ethereum", r"finance", r"financial", r"wealth",
    r"\b(make|earn|get)\s+\$?\d{3,}\b"
]
intent_pattern = re.compile("|".join(intent_keywords), re.I)

strategy_keywords = [
    r"affiliate marketing", r"clickbank", r"commission", r"online business",
    r"crypto", r"bitcoin", r"ethereum", r"defi", r"web3",
    r"day trading", r"options trading", r"swing trade",
    r"dropshipping", r"shopify", r"print on demand",
    r"\b(upwork|fiverr)\b", r"freelance client", r"remote gig"
]
strategy_pattern = re.compile("|".join(strategy_keywords), re.I)

negative_keywords = [
    r"instrumental", r"lyrics?", r"karaoke",
    r"minecraft", r"roblox", r"fortnite", r"gta",
    r"trailer", r"official video",
    r"breaking news", r"podcast", r"film", r"cinema"
]
negative_pattern = re.compile("|".join(negative_keywords), re.I)

# with input_path.open("r", encoding="utf-8") as fh:
#     total_lines = sum(1 for _ in fh)
# print(f"Total lines in file: {total_lines:,}")
total_lines = 72924794

sample_idx = set(random.sample(range(total_lines), sample_size))
batch_size = 10000
num_processes = mp.cpu_count()



def check_match(args):
    i, obj = args
    text = (obj.get("title", "") + " " + obj.get("description", "") + " " + obj.get("tags", "")).lower()
    if intent_pattern.search(text) and strategy_pattern.search(text) and not negative_pattern.search(text):
        return i, obj
    return None

def process_batch(batch, pool):
    results = pool.map(check_match, batch)
    return [r for r in results if r is not None]

def main():
    kept = 0
    batch = []

    with mp.Pool(num_processes) as pool:
        with jsonlines.open(input_path) as reader, jsonlines.open(output_all,"w") as writer_all, jsonlines.open(output_sample50,"w") as writer_sample50:
            for i, obj in enumerate(reader):
                batch.append((i, obj))
                if len(batch) >= batch_size or i == total_lines - 1:
                    matched_items = process_batch(batch, pool)
                    for idx, item in matched_items:
                        writer_all.write(item)
                        kept += 1
                        if idx in sample_idx:
                            writer_sample50.write(item)
                    batch = []


                if i % 100000 == 0:
                    print(f"\rProgress: {i/total_lines*100:.2f}%", end="")

    print(f"\nKept {kept:,} money-making lines → {output_all.name}")

if __name__ == '__main__':
    mp.freeze_support()
    main()