import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from llm_utils import add_backend_args, create_llm_from_args
from prompts.normalize_prompt import (
    NORMALIZATION_SYSTEM_PROMPT,
    build_normalization_prompt,
    extract_json_with_basic_repair,
    validate_normalization_output,
)

def safe_to_csv(df: pd.DataFrame, path: str):
    """Safely save a DataFrame, ensuring the parent directory exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def empty_records_columns(args) -> list[str]:
    """Columns expected in the detailed normalization output."""
    return [
        args.doc_id_col,
        args.order_col,
        args.party_col,
        args.speaker_col,
        args.speaker_label_col,
        args.claim_idx_col,
        args.quote_col,
        "normalization_raw",
        "normalization_json",
        "point",
    ]


def empty_points_columns() -> list[str]:
    """Columns expected in the flattened point output."""
    return [
        "document_id",
        "intervention_id",
        "party",
        "speaker",
        "speaker_label",
        "claim_idx",
        "quote",
        "point",
    ]


def read_csv_or_empty(path: str, label: str, required: bool = True) -> pd.DataFrame:
    """Read a CSV without allowing empty/missing files to crash the pipeline."""
    if not path or not os.path.exists(path):
        msg = f"[WARN] {label} CSV not found: {path}"
        if required:
            print(msg + " | using empty DataFrame", flush=True)
        return pd.DataFrame()

    if os.path.getsize(path) == 0:
        print(f"[WARN] {label} CSV is empty: {path} | using empty DataFrame", flush=True)
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"[WARN] {label} CSV has no parseable columns: {path} | using empty DataFrame", flush=True)
        return pd.DataFrame()
    except Exception as e:
        if required:
            print(f"[ERROR] Could not read {label} CSV: {path} | {e} | using empty DataFrame", flush=True)
            return pd.DataFrame()
        print(f"[WARN] Could not read optional {label} CSV: {path} | {e} | using empty DataFrame", flush=True)
        return pd.DataFrame()


def ensure_columns(df: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    """Ensure required columns exist so downstream code can continue safely."""
    if df.empty and len(df.columns) == 0:
        return pd.DataFrame(columns=columns)

    for col in columns:
        if col not in df.columns:
            print(f"[WARN] {label} missing column '{col}' | filling with empty strings", flush=True)
            df[col] = ""
    return df


def write_empty_outputs(output_csv: str, output_points_csv: str, args, reason: str) -> None:
    """Write valid empty outputs so the parent pipeline can continue."""
    print(f"[WARN] No normalization performed: {reason}", flush=True)
    safe_to_csv(pd.DataFrame(columns=empty_records_columns(args)), output_csv)
    safe_to_csv(pd.DataFrame(columns=empty_points_columns()), output_points_csv)
    print(
        f"[DEBUG] Wrote empty normalized outputs: records={output_csv} | points={output_points_csv}",
        flush=True,
    )


def first_row(obj):
    """Return a Series when lookup returns duplicate rows as a DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[0]
    return obj


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
        "claim_idx": row_dict.get(args.claim_idx_col),
        "quote": row_dict.get(args.quote_col, ""),
        "point": point,
    }]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--claims_csv", type=str, required=True)
    parser.add_argument("--debates_csv", type=str, required=True)
    parser.add_argument("--summaries_csv", type=str, required=True, help="Summaries from incr_summary.py")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--party", type=str, required=True, help="Party name for output filename")
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--quote_col", type=str, default="quote")
    parser.add_argument("--doc_id_col", type=str, default="document_id")
    parser.add_argument("--order_col", type=str, default="intervention_id")
    parser.add_argument("--party_col", type=str, default="party")
    parser.add_argument("--speaker_col", type=str, default="speaker")
    parser.add_argument("--speaker_label_col", type=str, default="speaker_label")
    parser.add_argument("--claim_idx_col", type=str, default="claim_idx")

    parser.add_argument("--text_col", type=str, default="speech")
    parser.add_argument("--summary_col", type=str, default="running_summary_after")

    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--add_timestamp", action="store_true", help="Add timestamp to output filenames")
    add_backend_args(parser)

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

        chunks.append(f"{speaker} ({party}): {text}")

    return "\n\n---\n\n".join(chunks)


def normalize_single_quote(
    llm,
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
        max_new_tokens=500,
        temperature=0.0,
    )

    print(f"[DEBUG_RAW_OUTPUT] {repr(raw_output[:200])}", flush=True)

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
    output_csv = os.path.join(args.output_dir, f"{args.party}_normalized_records{timestamp_str}.csv")
    output_points_csv = os.path.join(args.output_dir, f"{args.party}_normalized{timestamp_str}.csv")

    claims_df = read_csv_or_empty(args.claims_csv, "claims")
    debates_df = read_csv_or_empty(args.debates_csv, "debates")
    summaries_df = read_csv_or_empty(args.summaries_csv, "summaries", required=False)

    required_claim_cols = [args.doc_id_col, args.order_col, args.quote_col]
    required_debate_cols = [args.doc_id_col, args.order_col, args.text_col]
    optional_claim_cols = [args.party_col, args.speaker_col, args.speaker_label_col, args.claim_idx_col]
    optional_debate_cols = [args.party_col, args.speaker_col]
    required_summary_cols = [args.doc_id_col, args.order_col, args.summary_col]

    claims_df = ensure_columns(claims_df, required_claim_cols + optional_claim_cols, "claims")
    debates_df = ensure_columns(debates_df, required_debate_cols + optional_debate_cols, "debates")
    summaries_df = ensure_columns(summaries_df, required_summary_cols, "summaries")

    if claims_df.empty:
        write_empty_outputs(output_csv, output_points_csv, args, "claims CSV is empty or unreadable")
        return

    if debates_df.empty:
        print("[WARN] debates CSV is empty or unreadable | all claims will be saved with empty points", flush=True)

    # Coerce order IDs to numeric when possible, but do not crash on mixed types.
    for df_name, df in [("claims", claims_df), ("debates", debates_df), ("summaries", summaries_df)]:
        if args.order_col in df.columns:
            df[args.order_col] = pd.to_numeric(df[args.order_col], errors="ignore")

    sort_cols = [args.doc_id_col, args.order_col]
    if not debates_df.empty:
        debates_df = debates_df.sort_values(sort_cols).reset_index(drop=True)
    if not summaries_df.empty:
        summaries_df = summaries_df.sort_values(sort_cols).reset_index(drop=True)

    llm = create_llm_from_args(args)

    processed_records = []
    flattened_points = []

    start_idx = 0
    if args.resume and os.path.exists(output_csv):
        done_df = read_csv_or_empty(output_csv, "existing normalized records", required=False)
        if not done_df.empty:
            processed_records = done_df.to_dict("records")
            start_idx = min(len(done_df), len(claims_df))
            print(f"[DEBUG] Resume enabled | start_idx={start_idx}", flush=True)
        else:
            print("[WARN] Resume requested but existing output is empty/unreadable | starting from row 0", flush=True)

    debates_lookup = debates_df.set_index([args.doc_id_col, args.order_col]) if not debates_df.empty else pd.DataFrame()
    summaries_lookup = summaries_df.set_index([args.doc_id_col, args.order_col]) if not summaries_df.empty else pd.DataFrame()

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

        if debates_lookup.empty:
            print(f"[ERROR] No debate lookup available for doc_id={doc_id}, intervention_id={intervention_id}", flush=True)
            row_dict["normalization_raw"] = "ERROR: empty debate lookup"
            row_dict["normalization_json"] = json.dumps({"point": ""}, ensure_ascii=False)
            row_dict["point"] = ""
            processed_records.append(row_dict)
            continue

        try:
            debate_row = first_row(debates_lookup.loc[(doc_id, intervention_id)])
        except KeyError:
            print(f"[ERROR] Missing debate row for doc_id={doc_id}, intervention_id={intervention_id}", flush=True)
            row_dict["normalization_raw"] = "ERROR: missing debate row"
            row_dict["normalization_json"] = json.dumps({"point": ""}, ensure_ascii=False)
            row_dict["point"] = ""
            processed_records.append(row_dict)
            continue

        # Format current intervention with speaker and party
        speaker = str(debate_row.get(args.speaker_col, "")).strip()
        party = str(debate_row.get(args.party_col, "")).strip()
        intervention_text = str(debate_row.get(args.text_col, "")) if pd.notna(debate_row.get(args.text_col, "")) else ""
        intervention = f"{speaker} ({party}): {intervention_text}"
        
        # Safely extract summary from lookup
        summary = ""
        if not summaries_lookup.empty and (doc_id, intervention_id) in summaries_lookup.index:
            summary_row = first_row(summaries_lookup.loc[(doc_id, intervention_id)])
            summary_value = summary_row.get(args.summary_col, "")
            summary = str(summary_value) if pd.notna(summary_value) else ""

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

        flattened_points.extend(flatten_normalized_row(row_dict, parsed_output, args))

        # Incremental saving at checkpoints (safe for crashes)
        if (i + 1 - start_idx) % args.checkpoint_every == 0:
            checkpoint_records_df = pd.DataFrame(processed_records)
            checkpoint_points_df = pd.DataFrame(flattened_points)
            if checkpoint_points_df.empty:
                checkpoint_points_df = pd.DataFrame(columns=empty_points_columns())
            safe_to_csv(checkpoint_records_df, output_csv)
            print(f"[DEBUG] Checkpoint saved at row {i+1}", flush=True)
            safe_to_csv(checkpoint_points_df, output_points_csv)

    # Final save. Always write valid CSVs with headers, even when no points were produced.
    records_df = pd.DataFrame(processed_records)
    if records_df.empty:
        records_df = pd.DataFrame(columns=empty_records_columns(args))

    points_df = pd.DataFrame(flattened_points)
    if points_df.empty:
        points_df = pd.DataFrame(columns=empty_points_columns())

    safe_to_csv(records_df, output_csv)
    safe_to_csv(points_df, output_points_csv)
    print("[DEBUG] Normalization finished successfully.", flush=True)


if __name__ == "__main__":
    main()
