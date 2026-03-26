import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

from llm_utils import LocalLLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
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
            "normalized": claim.get("normalized", ""),
        })

    return flattened


def main():
    args = parse_args()

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    claims_output_dir = os.path.dirname(args.output_claims_csv)
    if claims_output_dir:
        os.makedirs(claims_output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = df.sort_values([args.doc_id_col, args.order_col]).reset_index(drop=True)

    llm = LocalLLM(args.model_name)

    processed_records = []
    flattened_claims = []

    start_idx = 0
    if args.resume and os.path.exists(args.output_csv):
        done_df = pd.read_csv(args.output_csv)
        start_idx = len(done_df)
        processed_records = done_df.to_dict("records")

        if os.path.exists(args.output_claims_csv):
            claims_df = pd.read_csv(args.output_claims_csv)
            flattened_claims = claims_df.to_dict("records")

    for i in tqdm(range(start_idx, len(df)), total=len(df) - start_idx):
        row = df.iloc[i]
        row_dict = row.to_dict()

        if args.target_party is not None and row[args.party_col] != args.target_party:
            row_dict["claim_extraction_raw"] = ""
            row_dict["claim_extraction_json"] = json.dumps({"claims": []}, ensure_ascii=False)
            row_dict["n_claims"] = 0
            processed_records.append(row_dict)
            continue

        summary = str(row[args.summary_col]) if pd.notna(row[args.summary_col]) else ""
        text = str(row[args.text_col]) if pd.notna(row[args.text_col]) else ""

        try:
            result = llm.extract_claims(summary=summary, intervention_text=text)
            parsed_output = result["parsed_output"]
            raw_output = result["raw_model_output"]
        except Exception as e:
            parsed_output = {"claims": []}
            raw_output = f"ERROR: {str(e)}"

        row_dict["claim_extraction_raw"] = raw_output
        row_dict["claim_extraction_json"] = json.dumps(parsed_output, ensure_ascii=False)
        row_dict["n_claims"] = len(parsed_output.get("claims", []))

        processed_records.append(row_dict)

        temp_flat = flatten_claims_row(row_dict, parsed_output, args)
        flattened_claims.extend(temp_flat)

        pd.DataFrame(processed_records).to_csv(args.output_csv, index=False)
        pd.DataFrame(flattened_claims).to_csv(args.output_claims_csv, index=False)

    pd.DataFrame(processed_records).to_csv(args.output_csv, index=False)
    pd.DataFrame(flattened_claims).to_csv(args.output_claims_csv, index=False)


if __name__ == "__main__":
    main()