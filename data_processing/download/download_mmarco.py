import os
import json
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# -----------------------------
# SETTINGS
# -----------------------------
languages = [
    "indonesian", "arabic", "japanese", "russian", "chinese",
    "french", "dutch", "german", "italian", "portuguese",
    "spanish", "hindi", "vietnamese"
]

# MSMARCO qrels (if available)
MSMARCO_QRELS_PATH = "data/msmarco/devs.tsv"

# Disable caching completely (optional)
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache_disabled"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# -----------------------------
# LOAD QRELS (optional)
# -----------------------------
if os.path.exists(MSMARCO_QRELS_PATH):
    msmarco_qrels = pd.read_csv(MSMARCO_QRELS_PATH, sep="\t")
    msmarco_qrels.columns = ["query-id", "corpus-id", "score"]
    print(f"Loaded MSMARCO qrels ({len(msmarco_qrels)} rows)")
else:
    msmarco_qrels = None
    print("⚠️ MSMARCO qrels not found — will create placeholder qrels.")

# -----------------------------
# PROCESS EACH LANGUAGE
# -----------------------------
for lang in languages:
    print(f"\n=== Processing {lang} ===")

    out_dir = f"data/mmarco/{lang}"
    os.makedirs(out_dir, exist_ok=True)

    # ---- Queries ----
    print("→ Loading all queries…")
    queries = load_dataset(
        "unicamp-dl/mmarco",
        f"queries-{lang}",
        split="dev",
        streaming=False,  # full dataset
        trust_remote_code=True
    )

    with open(f"{out_dir}/queries.jsonl", "w", encoding="utf-8") as f:
        for q in tqdm(queries, desc=f"Writing {lang} queries"):
            f.write(json.dumps({"_id": str(q["id"]), "text": q["text"]}, ensure_ascii=False) + "\n")

    # ---- Corpus ----
    print("→ Loading full corpus…")
    corpus = load_dataset(
        "unicamp-dl/mmarco",
        f"collection-{lang}",
        split="collection",
        streaming=False,  # full dataset
        trust_remote_code=True
    )

    with open(f"{out_dir}/corpus.jsonl", "w", encoding="utf-8") as f:
        for doc in tqdm(corpus, desc=f"Writing {lang} corpus"):
            f.write(json.dumps({"_id": str(doc["id"]), "text": doc["text"]}, ensure_ascii=False) + "\n")

    # ---- Qrels ----
    qrels_path = f"{out_dir}/qrels.tsv"
    if msmarco_qrels is not None:
        msmarco_qrels.to_csv(qrels_path, sep="\t", index=False)
    else:
        pd.DataFrame(columns=["query-id", "corpus-id", "score"]).to_csv(qrels_path, sep="\t", index=False)

    print(f"✅ Done: {lang} full dataset created")

print("\n🎯 All languages processed successfully (full dataset)!")
