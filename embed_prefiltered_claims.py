import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INPUT_ROOT = Path("/scratch_shared/lsaleh/prefilter_pipeline")
ALT_INPUT_ROOT = Path("/scratch-shared/lsaleh/prefilter_pipeline")
REQUIRED_INDEX_COLUMNS = [
    "document_id",
    "date",
    "datecount",
    "party",
    "claim_idx",
    "quote",
    "point",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed all prefiltered normalized claims for one party and save an "
            "embedding matrix plus an aligned metadata index."
        )
    )
    parser.add_argument("party", help="Party label, e.g. VVD, PVDA, GL.")
    parser.add_argument(
        "--input_root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root containing {party}/normalized CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for embedding outputs. Defaults to {input_root}/{party}/embeddings.",
    )
    parser.add_argument(
        "--debates_csv",
        type=Path,
        default=Path("outputs/debates.csv"),
        help="Debate metadata CSV used to fill date/datecount when missing.",
    )
    parser.add_argument(
        "--model_name",
        default=MODEL_NAME,
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument("--text_col", default="point", help="Column to embed.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Embedding device.",
    )
    parser.add_argument(
        "--allow_missing_dates",
        action="store_true",
        help="Write outputs even if date/datecount cannot be filled.",
    )
    return parser.parse_args()


def get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_input_root(input_root: Path) -> Path:
    if input_root.exists():
        return input_root
    if input_root == DEFAULT_INPUT_ROOT and ALT_INPUT_ROOT.exists():
        print(f"[WARN] {DEFAULT_INPUT_ROOT} not found; using {ALT_INPUT_ROOT}")
        return ALT_INPUT_ROOT
    return input_root


def cmp_rank_from_path(path: Path) -> int | None:
    match = re.search(r"_cmp_(\d+)_prefiltered_normalized\.csv$", path.name)
    return int(match.group(1)) if match else None


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_party_claims(party: str, input_root: Path) -> pd.DataFrame:
    normalized_dir = input_root / party / "normalized"
    paths = sorted(normalized_dir.glob(f"{party}_cmp_*_prefiltered_normalized.csv"))
    if not paths:
        raise FileNotFoundError(f"No normalized files found in {normalized_dir}")

    frames = []
    for path in paths:
        df = read_csv_or_empty(path)
        if df.empty and len(df.columns) == 0:
            print(f"[WARN] Empty or unreadable CSV skipped: {path}")
            continue
        df["source_file"] = str(path)
        df["cmp_rank"] = cmp_rank_from_path(path)
        frames.append(df)

    if not frames:
        raise ValueError(f"No readable normalized rows found in {normalized_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined["party"] = party
    return combined


def attach_date_metadata(df: pd.DataFrame, debates_csv: Path) -> pd.DataFrame:
    date_col = first_existing_column(df, ["date", "foi_meetingDate", "dc_date", "dc_date_year"])
    datecount_col = first_existing_column(df, ["datecount", "day_count", "date_count"])

    if date_col and date_col != "date":
        df["date"] = df[date_col]
    elif not date_col:
        df["date"] = pd.NA

    if datecount_col and datecount_col != "datecount":
        df["datecount"] = df[datecount_col]
    elif not datecount_col:
        df["datecount"] = pd.NA

    needs_merge = df["date"].isna().any() or df["datecount"].isna().any()
    if not needs_merge:
        return df

    if not debates_csv.exists():
        print(f"[WARN] Debate metadata not found, cannot fill dates: {debates_csv}")
        return df

    debates = pd.read_csv(debates_csv)
    if "dc_identifier" not in debates.columns:
        print(f"[WARN] Debate metadata missing dc_identifier: {debates_csv}")
        return df

    date_meta_col = first_existing_column(debates, ["foi_meetingDate", "dc_date", "dc_date_year"])
    datecount_meta_col = first_existing_column(debates, ["day_count", "datecount", "date_count"])
    keep_cols = ["dc_identifier"]
    rename = {"dc_identifier": "document_id"}
    if date_meta_col:
        keep_cols.append(date_meta_col)
        rename[date_meta_col] = "_meta_date"
    if datecount_meta_col:
        keep_cols.append(datecount_meta_col)
        rename[datecount_meta_col] = "_meta_datecount"

    meta = debates[keep_cols].drop_duplicates("dc_identifier").rename(columns=rename)
    df = df.merge(meta, on="document_id", how="left")

    if "_meta_date" in df.columns:
        df["date"] = df["date"].fillna(df["_meta_date"])
    if "_meta_datecount" in df.columns:
        df["datecount"] = df["datecount"].fillna(df["_meta_datecount"])

    return df.drop(columns=[c for c in ["_meta_date", "_meta_datecount"] if c in df.columns])


def validate_columns(df: pd.DataFrame, text_col: str, allow_missing_dates: bool) -> None:
    required = {"document_id", "party", "claim_idx", "quote", text_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in normalized claims: {missing}")

    if not allow_missing_dates:
        missing_date_rows = df["date"].isna().sum() + df["datecount"].isna().sum()
        if missing_date_rows:
            raise ValueError(
                "Could not fill date/datecount for all rows. "
                "Pass --allow_missing_dates to write outputs anyway."
            )


def main() -> None:
    args = parse_args()
    input_root = resolve_input_root(args.input_root)
    output_dir = args.output_dir or input_root / args.party / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_party_claims(args.party, input_root)
    df = attach_date_metadata(df, args.debates_csv)
    validate_columns(df, args.text_col, args.allow_missing_dates)

    df = df[df[args.text_col].notna()].copy()
    df[args.text_col] = df[args.text_col].astype(str).str.strip()
    df = df[df[args.text_col] != ""].reset_index(drop=True)
    df["embedding_id"] = df.index

    texts = df[args.text_col].tolist()
    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Rows to embed: {len(texts)}")
    print(f"Input root: {input_root}")
    print(f"Output dir: {output_dir}")

    model = SentenceTransformer(args.model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    index_cols = ["embedding_id", "cmp_rank", "source_file"] + REQUIRED_INDEX_COLUMNS
    extra_cols = [c for c in df.columns if c not in index_cols]
    index_df = df[index_cols + extra_cols]

    stem = f"{args.party}_prefiltered_claim_sbert"
    emb_path = output_dir / f"{stem}_embeddings.npy"
    index_path = output_dir / f"{stem}_embedding_index.csv"

    np.save(emb_path, embeddings)
    index_df.to_csv(index_path, index=False)

    print(f"Saved embeddings: {emb_path}")
    print(f"Shape: {embeddings.shape}")
    print(f"Saved index: {index_path}")


if __name__ == "__main__":
    main()
