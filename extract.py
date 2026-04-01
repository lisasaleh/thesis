import os
import json
import argparse
from typing import Dict, Any
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from llm_utils import LocalLLM, generate_json
from prompts.extract_prompt import build_extract_prompt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_claims_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--text_col", type=str, default="speech")
    parser.add_argument("--summary_col", type=str, default="summary_before")
    parser.add_argument("--party_col", type=str, default="party")
    parser.add_argument("--doc_id_col", type=str, default="document_id")
    parser.add_argument("--order_col", type=str, default="intervention_id")
    parser.add_argument("--speaker_col", type=str, default="speaker")
    parser.add_argument("--speaker_label_col", type=str, default="speaker_label")

    parser.add_argument("--target_party", type=str, default=None)
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def validate_claim_extraction_output(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"claims": []}

    claims = data.get("claims", [])
    if not isinstance(claims, list):
        return {"claims": []}

    cleaned_claims = []

    for item in claims:
        if not isinstance(item, dict):
            continue

        quote = item.get("quote", "")

        if isinstance(quote, str) and quote.strip():
            cleaned_claims.append({"quote": quote.strip()})

    return {"claims": cleaned_claims}


def extract_claims(llm, intervention_text: str, summary: str = ""):
    prompt = build_extract_prompt(
        interruption_text=intervention_text,
        debate_summary=summary,
    )

    parsed = generate_json(llm, prompt, max_new_tokens=300)
    parsed = validate_claim_extraction_output(parsed)

    return {
        "model_output_json": json.dumps(parsed, ensure_ascii=False),
        "parsed_output": parsed,
    }


def flatten_claims_row(row_dict, parsed_output, args):
    flattened = []
    claims = parsed_output.get("claims", [])

    for idx, claim in enumerate(claims):
        flattened.append({
            "document_id": row_dict.get(args.doc_id_col),
            "intervention_id": row_dict.get(args.order_col),
            "party": row_dict.get(args.party_col),
            "speaker": row_dict.get(args.speaker_col),
            "speaker_label": row_dict.get(args.speaker_label_col),
            "claim_idx": idx,
            "quote": claim.get("quote", ""),
        })

    return flattened


def main():
    args = parse_args()

    # Ensure output directories exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = args.target_party + args.input_csv.split("/")[-1].replace(".csv", f"_{timestamp}.csv")
    output_csv = base_name
    claims_name = base_name.replace(".csv", "_claims.csv")
    output_csv_path = os.path.join(args.output_dir, output_csv)
    output_claims_csv = os.path.join(args.output_dir, claims_name)

    # Load and sort data
    df = pd.read_csv(args.input_csv)
    df = df.sort_values([args.doc_id_col, args.order_col]).reset_index(drop=True)

    # Debug party info
    if args.target_party is not None:
        parties = sorted(
            df[args.party_col]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
        print(f"[DEBUG] target_party={args.target_party}", flush=True)
        print(f"[DEBUG] unique parties in data={parties}", flush=True)

    # Load model
    print("[DEBUG] Starting model load...", flush=True)
    llm = LocalLLM(args.model_name)
    print("[DEBUG] Model load finished.", flush=True)

    processed_records = []
    flattened_claims = []

    # Resume logic
    start_idx = 0
    if args.resume and os.path.exists(args.output_csv):
        done_df = pd.read_csv(args.output_csv)
        start_idx = len(done_df)
        processed_records = done_df.to_dict("records")

        if os.path.exists(args.output_claims_csv):
            claims_df = pd.read_csv(args.output_claims_csv)
            flattened_claims = claims_df.to_dict("records")

        print(f"[DEBUG] Resume enabled | start_idx={start_idx}", flush=True)

    target_party = (
        args.target_party.strip().lower()
        if args.target_party is not None
        else None
    )

    # Main loop
    for i in tqdm(range(start_idx, len(df)), total=len(df) - start_idx):
        row = df.iloc[i]
        row_dict = row.to_dict()

        row_party = (
            str(row[args.party_col]).strip().lower()
            if pd.notna(row[args.party_col])
            else ""
        )

        # Skip non-target parties
        if target_party is not None and row_party != target_party:
            row_dict["claim_extraction_raw"] = ""
            row_dict["claim_extraction_json"] = json.dumps({"claims": []}, ensure_ascii=False)
            row_dict["n_claims"] = 0
            processed_records.append(row_dict)
            continue

        summary = str(row[args.summary_col]) if pd.notna(row[args.summary_col]) else ""
        text = str(row[args.text_col]) if pd.notna(row[args.text_col]) else ""

        # Skip empty text
        if not text.strip():
            row_dict["claim_extraction_raw"] = ""
            row_dict["claim_extraction_json"] = json.dumps({"claims": []}, ensure_ascii=False)
            row_dict["n_claims"] = 0
            processed_records.append(row_dict)
            continue

        print(
            f"[DEBUG] Processing row={i} | doc_id={row_dict.get(args.doc_id_col)} | intervention_id={row_dict.get(args.order_col)}",
            flush=True,
        )

        try:
            result = extract_claims(
                llm,
                intervention_text=text,
                summary=summary,
            )
            parsed_output = result["parsed_output"]
            model_output_json = result["model_output_json"]

        except Exception as e:
            print(f"[ERROR] Failed on row {i}: {e}", flush=True)
            raise

        # Store results
        row_dict["claim_extraction_raw"] = model_output_json
        row_dict["claim_extraction_json"] = json.dumps(parsed_output, ensure_ascii=False)
        row_dict["n_claims"] = len(parsed_output.get("claims", []))

        processed_records.append(row_dict)

        # Flatten claims
        temp_flat = flatten_claims_row(row_dict, parsed_output, args)
        flattened_claims.extend(temp_flat)

        # Incremental saving (safe for crashes)
        pd.DataFrame(processed_records).to_csv(args.output_csv, index=False)
        pd.DataFrame(flattened_claims).to_csv(args.output_claims_csv, index=False)

    # Final save
    pd.DataFrame(processed_records).to_csv(args.output_csv, index=False)
    pd.DataFrame(flattened_claims).to_csv(args.output_claims_csv, index=False)

    print("[DEBUG] Extraction finished successfully.", flush=True)


if __name__ == "__main__":
    main()