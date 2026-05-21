# scripts/embed_debate_claims_sbert.py

from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


INPUT_PATH = Path("outputs/samples/VVD_cmp_1_labeled_true.csv")  # change if needed
OUTPUT_DIR = Path("outputs/embeddings/debates")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_COL = "point"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    df = pd.read_csv(INPUT_PATH)

    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing text column: {TEXT_COL}")

    df = df[df[TEXT_COL].notna()].copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL] != ""].copy()

    df = df.reset_index(drop=True)
    df["embedding_id"] = df.index

    texts = df[TEXT_COL].tolist()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Rows to embed: {len(texts)}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    stem = INPUT_PATH.stem

    emb_path = OUTPUT_DIR / f"{stem}_sbert_embeddings.npy"
    index_path = OUTPUT_DIR / f"{stem}_sbert_embedding_index.csv"

    np.save(emb_path, embeddings)
    df.to_csv(index_path, index=False)

    print(f"Saved embeddings: {emb_path}")
    print(f"Shape: {embeddings.shape}")
    print(f"Saved index: {index_path}")


if __name__ == "__main__":
    main()