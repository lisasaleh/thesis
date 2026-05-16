import os
import re
import argparse
import shutil
import hashlib
from pathlib import Path

# Transformers stores trusted remote-code modules under a generated path that
# can exceed Windows path limits for this ManifestoBERT repo. Use a short cache
# path before importing transformers.
os.environ.setdefault("HF_MODULES_CACHE", r"C:\hf\modules")

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "outputs/samples/VVD_cmp_1_normalized_full.csv"
MANIFEST_FILE = "outputs/cmp_manifest.csv"
OUTPUT_FILE = "outputs/samples/VVD_cmp_1_labeled_full.csv"

MODEL_NAME = (
    "manifesto-project/"
    "manifestoberta-xlm-roberta-56policy-topics-sentence-2024-1-1"
)
TOKENIZER_NAME = "xlm-roberta-large"

# ============================================================
# HELPERS
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Label normalized claims with ManifestoBERT CMP codes.")
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--manifest_file", default=MANIFEST_FILE)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--tokenizer_name", default=TOKENIZER_NAME)
    parser.add_argument("--text_col", default="point")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def get_party_and_rank_from_filename(filename: str):
    """
    Example:
        VVD cmp_1.csv

    Returns:
        ("VVD", 1)
    """

    # operate on the basename (remove path and extension)
    base = os.path.basename(filename)
    base = os.path.splitext(base)[0]

    match = re.search(
        r"(.+?)\s*cmp[_\s-]*(\d+)",
        base,
        re.IGNORECASE
    )

    if not match:
        raise ValueError(
            f"Could not parse party/rank from filename: {filename}"
        )

    # remove trailing separators from the extracted party name
    party = re.sub(r"[_\s-]+$", "", match.group(1).strip())
    rank = int(match.group(2))

    return party, rank



def get_target_cmp_code(filename: str, manifest_df: pd.DataFrame):
    """
    Example:
        VVD_cmp_1.csv

    Looks up:
        party == VVD
        code_1

    Returns:
        605
    """

    party, rank = get_party_and_rank_from_filename(filename)

    row = manifest_df.loc[
        manifest_df["party"].astype(str).str.strip().eq(party)
    ]

    if row.empty:
        raise ValueError(f"Party {party!r} not found")

    code_col = f"code_{rank}"

    if code_col not in manifest_df.columns:
        raise ValueError(f"Column {code_col!r} not found")

    target_code = str(row.iloc[0][code_col]).strip()

    return party, rank, target_code



def normalize_cmp_code(label: str):
    """
    ManifestoBERT sometimes returns labels like:

        per605 or 605 - Law and Order: Positive

    Convert to:

        605
    """

    label = str(label)
    label = label.strip()
    # Handles labels such as "605", "per605", or "605 - Law and Order: Positive".
    match = re.search(r"(\d{3})", label)
    if match:
        return match.group(1)
    return ""


def get_target_label_id(model, target_code: str):
    target_code = str(target_code).strip()
    for id_val, label_str in model.config.id2label.items():
        if normalize_cmp_code(label_str) == target_code:
            return int(id_val)
    return None


def classify_points(df: pd.DataFrame, tokenizer, model, target_code: str, text_col: str, batch_size: int, max_length: int):
    device = next(model.parameters()).device
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    target_id = get_target_label_id(model, target_code)

    texts = df[text_col].fillna("").astype(str).tolist()
    predicted_codes = []
    confidences = []
    target_confidences = []
    target_ranks = []
    top3_codes = []
    top3_confidences = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).cpu()
        pred_ids = probs.argmax(dim=-1).tolist()
        top_probs, top_ids = torch.topk(probs, k=min(3, probs.shape[1]), dim=-1)

        for row_idx, pred_id in enumerate(pred_ids):
            predicted_codes.append(normalize_cmp_code(id2label[pred_id]))
            confidences.append(float(probs[row_idx, pred_id].item()))
            top_codes = [normalize_cmp_code(id2label[int(i)]) for i in top_ids[row_idx].tolist()]
            top_scores = [float(x) for x in top_probs[row_idx].tolist()]
            top3_codes.append("|".join(top_codes))
            top3_confidences.append("|".join(f"{x:.6f}" for x in top_scores))

            if target_id is None:
                target_confidences.append(0.0)
                target_ranks.append(None)
            else:
                target_confidences.append(float(probs[row_idx, target_id].item()))
                sorted_ids = torch.argsort(probs[row_idx], descending=True).tolist()
                target_ranks.append(sorted_ids.index(target_id) + 1)

    return predicted_codes, confidences, target_confidences, target_ranks, top3_codes, top3_confidences


def clear_dynamic_module_cache(model_name: str):
    cache_roots = [
        Path(os.environ.get("HF_MODULES_CACHE", r"C:\hf\modules")),
        Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules",
    ]

    org, _, repo = model_name.partition("/")
    candidates = []
    for cache_root in cache_roots:
        candidates.extend([
            cache_root / org / repo,
            cache_root / org.replace("-", "_hyphen_") / repo.replace("-", "_hyphen_"),
        ])

    for path in candidates:
        if path.exists():
            print(f"Clearing stale Hugging Face dynamic module cache: {path}")
            shutil.rmtree(path)


def load_model_and_tokenizer(model_name: str, tokenizer_name: str | None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)

    def load_model(force_download: bool = False):
        loaded = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            force_download=force_download,
            output_loading_info=True,
        )
        model, loading_info = loaded
        missing = loading_info.get("missing_keys", [])
        unexpected = loading_info.get("unexpected_keys", [])
        if missing:
            print(f"[WARN] Missing model weights: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        if unexpected:
            print(f"[WARN] Unexpected model weights: {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")
        critical_missing = [k for k in missing if "final_classifier" in k or "classifier" in k]
        if critical_missing:
            raise RuntimeError(
                "Classification head weights were not loaded correctly: "
                + ", ".join(critical_missing)
            )
        return model

    try:
        model = load_model()
    except FileNotFoundError as e:
        missing_path = str(e)
        if "transformers_modules" not in missing_path:
            raise

        print("Detected stale/incomplete Hugging Face dynamic module cache.")
        clear_dynamic_module_cache(model_name)
        try:
            model = load_model(force_download=True)
        except FileNotFoundError as retry_error:
            retry_missing_path = str(retry_error)
            if "transformers_modules" not in retry_missing_path:
                raise

            # Some Transformers versions fail on Windows while copying a remote
            # custom module if the commit-specific destination folder is absent.
            match = re.search(r"No such file or directory: '([^']+)'", retry_missing_path)
            if not match:
                raise

            destination = Path(match.group(1))
            print(f"Creating missing dynamic module directory: {destination.parent}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            model = load_model(force_download=True)

    return tokenizer, model


def main():
    args = parse_args()

    print("Loading CSV files...")
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    if not os.path.exists(args.manifest_file):
        raise FileNotFoundError(f"Manifest file not found: {args.manifest_file}")

    manifest_df = pd.read_csv(args.manifest_file)
    df = pd.read_csv(args.input_file)

    if args.text_col not in df.columns:
        raise ValueError(f"Text column {args.text_col!r} not found. Available columns: {list(df.columns)}")

    df = df.reset_index(drop=True)
    df["source_row_id"] = df.index
    df["text_hash"] = df[args.text_col].fillna("").astype(str).apply(
        lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[:12]
    )

    party, cmp_rank, target_cmp_code = get_target_cmp_code(args.input_file, manifest_df)

    print(f"Party: {party}")
    print(f"CMP Rank: {cmp_rank}")
    print(f"Target CMP Code: {target_cmp_code}")
    print(f"Rows: {len(df)}")

    print("Loading model...")
    tokenizer, model = load_model_and_tokenizer(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model device: {device}")

    print("Running classification...")
    (
        predicted_codes,
        confidences,
        target_confidences,
        target_ranks,
        top3_codes,
        top3_confidences,
    ) = classify_points(
        df=df,
        tokenizer=tokenizer,
        model=model,
        target_code=target_cmp_code,
        text_col=args.text_col,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    df["predicted_cmp_code"] = predicted_codes
    df["cmp_confidence"] = confidences
    df["target_code_confidence"] = target_confidences
    df["target_code_rank"] = target_ranks
    df["top3_cmp_codes"] = top3_codes
    df["top3_cmp_confidences"] = top3_confidences
    df["target_cmp_code"] = target_cmp_code
    df["party_from_filename"] = party
    df["cmp_rank_from_filename"] = cmp_rank
    df["matches_target_cmp"] = df["predicted_cmp_code"].astype(str) == df["target_cmp_code"].astype(str)
    df["target_in_top3"] = df["top3_cmp_codes"].apply(
        lambda codes: str(target_cmp_code) in str(codes).split("|")
    )

    expected_len = len(df)
    actual_len = len(predicted_codes)
    if actual_len != expected_len:
        raise RuntimeError(f"Prediction alignment error: got {actual_len} predictions for {expected_len} rows")

    df = df.sort_values(by="cmp_confidence", ascending=False)

    print(df[[
        args.text_col,
        "predicted_cmp_code",
        "cmp_confidence",
        "target_code_confidence",
        "target_code_rank",
        "top3_cmp_codes",
        "target_in_top3",
        "matches_target_cmp",
    ]].head())

    print("\nPredicted CMP distribution:")
    print(df["predicted_cmp_code"].value_counts().head(20))
    print("\nTarget code rank summary:")
    print(df["target_code_rank"].describe())
    print("\nTarget match rates:")
    print(f"Top-1 exact match: {df['matches_target_cmp'].mean():.3%}")
    print(f"Top-3 contains target: {df['target_in_top3'].mean():.3%}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_csv(args.output_file, index=False)
    print(f"Saved output to: {args.output_file}")


if __name__ == "__main__":
    main()
