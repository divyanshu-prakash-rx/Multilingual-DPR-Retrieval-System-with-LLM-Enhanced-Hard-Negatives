import json
import os
import datasets
import pandas as pd
from tqdm.auto import tqdm

split = "dev"

languages = [
    "indonesian",
    "arabic",
    "japanese",
    "russian",
    "chinese",
    "french",
    "dutch",
    "german",
    "italian",
    "portuguese",
    "spanish",
    "hindi",
    "vietnamese",
]

print("=== Downloading MMARCO ===")

for i, language in enumerate(languages):
    print(f"Language: {language}")
    print(f"Language {i+1} of {len(languages)}")

    # Load query and corpus datasets safely
    query_dataset = datasets.load_dataset("unicamp-dl/mmarco", "queries-" + language)
    corpus_dataset = datasets.load_dataset(
        "unicamp-dl/mmarco", "collection-" + language, trust_remote_code=True
    )
    corpus_dataset = corpus_dataset["collection"]

    print("Loaded corpus of size", len(corpus_dataset))
    print("---------------------------------------")

    # ---- QRELS / DEV SET ----
    qrels_path = f"data/msmarco/devs.tsv"
    if os.path.exists(qrels_path):
        qrels_df = pd.read_csv(qrels_path, sep="\t")
    else:
        # fallback: generate mock qrels if not available
        print(f"⚠️ Qrels file not found at {qrels_path}, generating placeholder file.")
        qrels_df = pd.DataFrame(columns=["query-id", "corpus-id", "score"])

    # Convert to BEIR-compatible format
    qrels_df = qrels_df.rename(columns={
        "query-id": "qid",
        "corpus-id": "docid",
        "score": "relevance"
    })

    # Save qrels
    os.makedirs(f"data/mmarco/beir/{language}/qrels", exist_ok=True)
    qrels_df.to_csv(
        f"data/mmarco/beir/{language}/qrels/{split}.tsv",
        sep="\t", index=False
    )

    # ---- QUERIES ----
    queries_dict = {"_id": [], "text": []}
    for query in tqdm(query_dataset[split], desc=f"Processing {language} queries"):
        queries_dict["_id"].append(str(query["id"]))
        queries_dict["text"].append(query["text"])
    query_df = pd.DataFrame(queries_dict)
    query_df.to_json(
        f"data/mmarco/beir/{language}/queries.jsonl", orient="records", lines=True
    )

    # ---- CORPUS ----
    # rename id fields safely
    if "id" in corpus_dataset.column_names:
        corpus_dataset = corpus_dataset.rename_column("id", "_id")
    elif "doc_id" in corpus_dataset.column_names:
        corpus_dataset = corpus_dataset.rename_column("doc_id", "_id")

    corpus_dataset = corpus_dataset.cast(
        datasets.Features({
            "_id": datasets.Value("string"),
            "text": datasets.Value("string")
        })
    )

    corpus_dataset.to_json(
        f"data/mmarco/beir/{language}/corpus.jsonl", orient="records", lines=True
    )

print("=== MMARCO download complete ===")
