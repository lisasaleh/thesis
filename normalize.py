import os
import json
import argparse
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from llm_utils import LocalLLM
from prompts.normalize_prompt import (
    NORMALIZATION_SYSTEM_PROMPT,
    build_normalization_prompt,
    extract_json_with_basic_repair,
    validate_normalization_output,
)

def flatten_normalized_row(row_dict, parsed_output, args):
    point = parsed_output.get("point", "").strip()

    if not point:
        return []

    return [{
        "document_id": row_dict.get(args.doc_id_col),
        "intervention_id": row_dict.get(args.order_col),
        "party": row_dict.get(args.party_col),
        "speaker": row_dict.get(args.speaker_col),
        "speaker_label": row_dict.get(args.speaker_label_col),
        "claim_idx": row_dict.get("claim_idx"),
        "quote": row_dict.get("quote", ""),
        "point": point,
    }]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--claims_csv", type=str, required=True)
    parser.add_argument("--debates_csv", type=str, required=True)
    parser.add_argument("--summaries_csv", type=str, required=True, help="Summaries from incr_summary.py")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--quote_col", type=str, default="quote")
    parser.add_argument("--doc_id_col", type=str, default="document_id")
    parser.add_argument("--order_col", type=str, default="intervention_id")
    parser.add_argument("--party_col", type=str, default="party")
    parser.add_argument("--speaker_col", type=str, default="speaker")
    parser.add_argument("--speaker_label_col", type=str, default="speaker_label")
    parser.add_argument("--claim_idx_col", type=str, default="claim_idx")

    parser.add_argument("--text_col", type=str, default="speech")
    parser.add_argument("--summary_col", type=str, default="summary_before")

    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--add_timestamp", action="store_true", help="Add timestamp to output filenames")

    return parser.parse_args()


def build_previous_interventions_text(
    debates_df: pd.DataFrame,
    doc_id,
    intervention_id,
    text_col: str,
    order_col: str,
    speaker_col: str,
    party_col: str,
    doc_id_col: str,
) -> str:
    doc_df = debates_df[debates_df[doc_id_col] == doc_id].sort_values(order_col)

    prev_rows = doc_df[doc_df[order_col] < intervention_id].tail(2)

    chunks = []
    for _, row in prev_rows.iterrows():
        speaker = str(row.get(speaker_col, "")).strip()
        party = str(row.get(party_col, "")).strip()
        text = str(row.get(text_col, "")).strip()

        chunks.append(f"Spreker: {speaker} ({party})\n{text}")

    return "\n\n---\n\n".join(chunks)


def normalize_single_quote(
    llm: LocalLLM,
    quote: str,
    intervention: str,
    summary: str,
    previous_interventions: str,
) -> dict:
    user_prompt = build_normalization_prompt(
        quote=quote,
        intervention=intervention,
        summary=summary,
        previous_interventions=previous_interventions,
    )

    raw_output = llm.generate(
        prompt=user_prompt,
        system_prompt=NORMALIZATION_SYSTEM_PROMPT,
        max_new_tokens=120,
        temperature=0.0,
    )

    parsed = extract_json_with_basic_repair(raw_output)
    validated = validate_normalization_output(parsed)

    return {
        "raw_model_output": raw_output,
        "parsed_output": validated,
    }


def main():
    args = parse_args()

    # Ensure output directories exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Generate output CSV with optional timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if args.add_timestamp else ""
    timestamp_str = f"_{timestamp}" if timestamp else ""
    base_name = args.claims_csv.split("/")[-1].replace(".csv", f"_normalized{timestamp_str}.csv")
    output_csv = os.path.join(args.output_dir, base_name)
    output_points_csv = os.path.join(args.output_dir, base_name.replace("_normalized", "_normalized_points"))

    claims_df = pd.read_csv(args.claims_csv)
    debates_df = pd.read_csv(args.debates_csv)
    summaries_df = pd.read_csv(args.summaries_csv)

    debates_df = debates_df.sort_values([args.doc_id_col, args.order_col]).reset_index(drop=True)
    summaries_df = summaries_df.sort_values([args.doc_id_col, args.order_col]).reset_index(drop=True)

    print("[DEBUG] Starting model load...", flush=True)
    llm = LocalLLM(args.model_name)
    print("[DEBUG] Model load finished.", flush=True)

    processed_records = []
    flattened_points = []

    start_idx = 0
    if args.resume and os.path.exists(output_csv):
        done_df = pd.read_csv(output_csv)
        processed_records = done_df.to_dict("records")
        start_idx = len(done_df)
        print(f"[DEBUG] Resume enabled | start_idx={start_idx}", flush=True)

    debates_lookup = debates_df.set_index([args.doc_id_col, args.order_col])
    summaries_lookup = summaries_df.set_index([args.doc_id_col, args.order_col])

    for i in tqdm(range(start_idx, len(claims_df)), total=len(claims_df) - start_idx):
        row = claims_df.iloc[i]
        row_dict = row.to_dict()

        doc_id = row[args.doc_id_col]
        intervention_id = row[args.order_col]
        quote = str(row[args.quote_col]) if pd.notna(row[args.quote_col]) else ""

        if not quote.strip():
            row_dict["normalization_raw"] = ""
            row_dict["normalization_json"] = json.dumps({"point": ""}, ensure_ascii=False)
            row_dict["point"] = ""
            processed_records.append(row_dict)
            continue

        try:
            debate_row = debates_lookup.loc[(doc_id, intervention_id)]
        except KeyError:
            print(f"[ERROR] Missing debate row for doc_id={doc_id}, intervention_id={intervention_id}", flush=True)
            row_dict["normalization_raw"] = "ERROR: missing debate row"
            row_dict["normalization_json"] = json.dumps({"point": ""}, ensure_ascii=False)
            row_dict["point"] = ""
            processed_records.append(row_dict)
            continue

        intervention = str(debate_row[args.text_col]) if pd.notna(debate_row[args.text_col]) else ""
        summary = str(summaries_lookup.loc[(doc_id, intervention_id)]["running_summary_after"]) if pd.notna(summaries_lookup.loc[(doc_id, intervention_id)]["running_summary_after"]) else ""

        previous_interventions = build_previous_interventions_text(
            debates_df=debates_df,
            doc_id=doc_id,
            intervention_id=intervention_id,
            text_col=args.text_col,
            order_col=args.order_col,
            speaker_col=args.speaker_col,
            party_col=args.party_col,
            doc_id_col=args.doc_id_col,
        )

        print(
            f"[DEBUG] Normalizing row={i} | doc_id={doc_id} | intervention_id={intervention_id} | claim_idx={row.get(args.claim_idx_col, '')}",
            flush=True
        )

        try:
            result = normalize_single_quote(
                llm=llm,
                quote=quote,
                intervention=intervention,
                summary=summary,
                previous_interventions=previous_interventions,
            )
            parsed_output = result["parsed_output"]
            raw_output = result["raw_model_output"]
        except Exception as e:
            print(f"[ERROR] Failed on row {i}: {e}", flush=True)
            parsed_output = {"point": ""}
            raw_output = f"ERROR: {str(e)}"

        row_dict["normalization_raw"] = raw_output
        row_dict["normalization_json"] = json.dumps(parsed_output, ensure_ascii=False)
        row_dict["point"] = parsed_output.get("point", "")

        processed_records.append(row_dict)

        temp_flat = flatten_normalized_row(row_dict, parsed_output, args)
        flattened_points.extend(temp_flat)

        # Incremental saving at checkpoints (safe for crashes)
        if (i + 1 - start_idx) % args.checkpoint_every == 0:
            pd.DataFrame(processed_records).to_csv(output_csv, index=False)
            print(f"[DEBUG] Checkpoint saved at row {i+1}", flush=True)
            pd.DataFrame(flattened_points).to_csv(output_points_csv, index=False)

    # Final save
    pd.DataFrame(processed_records).to_csv(output_csv, index=False)
    pd.DataFrame(flattened_points).to_csv(output_points_csv, index=False)
    print("[DEBUG] Normalization finished successfully.", flush=True)


if __name__ == "__main__":
    main()