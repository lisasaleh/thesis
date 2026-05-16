import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_NAME = (
    "manifesto-project/"
    "manifestoberta-xlm-roberta-56policy-topics-sentence-2024-1-1"
)
DEFAULT_TOKENIZER_NAME = "xlm-roberta-large"


def read_csv_or_empty(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_cmp_code(label: Any) -> str:
    label = str(label).strip()
    match = re.search(r"(\d{3,4})", label, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def get_party_and_rank_from_filename(filename: str) -> tuple[str, int]:
    base = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"(.+?)\s*cmp[_\s-]*(\d+)", base, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse party/rank from filename: {filename}")
    party = re.sub(r"[_\s-]+$", "", match.group(1).strip())
    return party, int(match.group(2))


def get_target_cmp_code(
    input_file: str,
    manifest_file: str,
    party: str | None,
    cmp_rank: int | None,
    target_cmp_code: str | None,
) -> tuple[str, int, str]:
    if target_cmp_code is not None:
        resolved_party = party or ""
        resolved_rank = cmp_rank or 0
        resolved_code = normalize_cmp_code(target_cmp_code)
        if not resolved_code:
            raise ValueError(f"Could not normalize target CMP code: {target_cmp_code!r}")
        return resolved_party, resolved_rank, resolved_code

    resolved_party = party
    resolved_rank = cmp_rank
    if resolved_party is None or resolved_rank is None:
        parsed_party, parsed_rank = get_party_and_rank_from_filename(input_file)
        resolved_party = resolved_party or parsed_party
        resolved_rank = resolved_rank or parsed_rank

    manifest_df = pd.read_csv(manifest_file)
    row = manifest_df.loc[
        manifest_df["party"].astype(str).str.strip().eq(str(resolved_party).strip())
    ]
    if row.empty:
        raise ValueError(f"Party {resolved_party!r} not found in {manifest_file}")

    code_col = f"code_{resolved_rank}"
    if code_col not in manifest_df.columns:
        raise ValueError(f"Column {code_col!r} not found in {manifest_file}")

    target_code = normalize_cmp_code(row.iloc[0][code_col])
    if not target_code:
        raise ValueError(f"Could not normalize target CMP code from {code_col}: {row.iloc[0][code_col]!r}")
    return str(resolved_party), int(resolved_rank), target_code


def label_metadata(model) -> list[dict[str, Any]]:
    labels = []
    for idx in range(model.config.num_labels):
        raw_label = model.config.id2label[idx]
        labels.append({
            "id": idx,
            "label": raw_label,
            "cmp_code": normalize_cmp_code(raw_label),
        })
    return labels


def classify_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    top_k: int,
    max_length: int,
) -> list[dict[str, Any]]:
    labels = label_metadata(model)
    results = []

    for start in tqdm(range(0, len(texts), batch_size), desc="RoBERTa CMP labeling"):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu()

        k = min(top_k, probs.shape[-1])
        top_probs, top_ids = torch.topk(probs, k=k, dim=-1)

        for row_idx in range(probs.shape[0]):
            row_probs = probs[row_idx]
            full_probs = {
                labels[label_idx]["cmp_code"]: float(row_probs[label_idx].item())
                for label_idx in range(len(labels))
            }
            top_items = []
            for label_id, prob in zip(top_ids[row_idx].tolist(), top_probs[row_idx].tolist()):
                top_items.append({
                    "rank": len(top_items) + 1,
                    "label_id": int(label_id),
                    "label": labels[label_id]["label"],
                    "cmp_code": labels[label_id]["cmp_code"],
                    "probability": float(prob),
                })

            results.append({
                "top_items": top_items,
                "full_probabilities": full_probs,
            })

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ManifestoBERT/RoBERTa CMP labeling and recall-oriented top-k prefiltering."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--manifest_file", default="outputs/cmp_manifest.csv")
    parser.add_argument("--party", default=None)
    parser.add_argument("--cmp_rank", type=int, default=None)
    parser.add_argument("--target_cmp_code", default=None)
    parser.add_argument("--text_col", default="quote")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer_name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.output_csv) and not args.force:
        print(f"[SKIPPED] Output exists: {args.output_csv}")
        return

    party, cmp_rank, target_cmp_code = get_target_cmp_code(
        input_file=args.input_csv,
        manifest_file=args.manifest_file,
        party=args.party,
        cmp_rank=args.cmp_rank,
        target_cmp_code=args.target_cmp_code,
    )

    df = read_csv_or_empty(args.input_csv)
    if df.empty and args.text_col not in df.columns:
        df = pd.DataFrame(columns=[args.text_col])
    elif args.text_col not in df.columns:
        raise ValueError(f"Text column {args.text_col!r} not found in {args.input_csv}")

    base_audit_columns = {
        "target_cmp_code": str(target_cmp_code),
        "cmp_prefilter_party": party,
        "cmp_prefilter_rank": cmp_rank,
        "cmp_roberta_model": args.model_name,
        "cmp_roberta_text_col": args.text_col,
        "cmp_roberta_top_k": args.top_k,
        "cmp_roberta_top_codes": "[]",
        "cmp_roberta_top_labels": "[]",
        "cmp_roberta_top_probs": "[]",
        "cmp_roberta_top_items_json": "[]",
        "cmp_roberta_all_probabilities_json": "{}",
        "cmp_roberta_target_probability": 0.0,
        "kept_by_prefilter": False,
    }

    if df.empty:
        for col, value in base_audit_columns.items():
            df[col] = value
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved empty labeled file to {args.output_csv} | target_cmp_code={target_cmp_code}")
        return

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    model_codes = {item["cmp_code"] for item in label_metadata(model)}
    if target_cmp_code not in model_codes:
        raise ValueError(
            f"Target CMP code {target_cmp_code!r} is not present in model labels. "
            f"Example labels: {sorted(code for code in model_codes if code)[:10]}"
        )

    texts = df[args.text_col].fillna("").astype(str).tolist()
    predictions = classify_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        max_length=args.max_length,
    )

    top_codes = []
    top_labels = []
    top_probs = []
    target_probs = []
    kept = []
    full_probs_json = []
    top_items_json = []

    for pred in predictions:
        items = pred["top_items"]
        codes = [item["cmp_code"] for item in items]
        labels = [item["label"] for item in items]
        probs = [item["probability"] for item in items]
        full_probs = pred["full_probabilities"]

        top_codes.append(json.dumps(codes, ensure_ascii=False))
        top_labels.append(json.dumps(labels, ensure_ascii=False))
        top_probs.append(json.dumps(probs, ensure_ascii=False))
        target_probs.append(float(full_probs.get(str(target_cmp_code), 0.0)))
        kept.append(str(target_cmp_code) in codes)
        full_probs_json.append(json.dumps(full_probs, ensure_ascii=False))
        top_items_json.append(json.dumps(items, ensure_ascii=False))

    df["target_cmp_code"] = str(target_cmp_code)
    df["cmp_prefilter_party"] = party
    df["cmp_prefilter_rank"] = cmp_rank
    df["cmp_roberta_model"] = args.model_name
    df["cmp_roberta_text_col"] = args.text_col
    df["cmp_roberta_top_k"] = args.top_k
    df["cmp_roberta_top_codes"] = top_codes
    df["cmp_roberta_top_labels"] = top_labels
    df["cmp_roberta_top_probs"] = top_probs
    df["cmp_roberta_top_items_json"] = top_items_json
    df["cmp_roberta_all_probabilities_json"] = full_probs_json
    df["cmp_roberta_target_probability"] = target_probs
    df["kept_by_prefilter"] = kept

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(
        f"Saved {len(df)} labeled rows to {args.output_csv} | "
        f"kept={int(sum(kept))} | target_cmp_code={target_cmp_code} | top_k={args.top_k}"
    )


if __name__ == "__main__":
    main()
