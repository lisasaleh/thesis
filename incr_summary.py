import argparse
import json
import os
from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from llm_utils import add_backend_args, create_llm_from_args, generate_json_with_raw
from prompts.summary_prompt import build_incremental_summary_prompt


def safe_to_csv(df: pd.DataFrame, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False)


def validate_state(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        return {
            "main_topic": "",
            "points_of_discussion": [],
            "updated_summary": "",
        }

    if "main_topic" not in parsed or not isinstance(parsed["main_topic"], str):
        parsed["main_topic"] = ""

    if "points_of_discussion" not in parsed or not isinstance(parsed["points_of_discussion"], list):
        parsed["points_of_discussion"] = []

    cleaned_points = []
    for item in parsed["points_of_discussion"]:
        if not isinstance(item, dict):
            continue

        point = item.get("point", "")
        arguments = item.get("arguments", [])

        if not isinstance(point, str):
            point = str(point)

        if not isinstance(arguments, list):
            arguments = []

        cleaned_arguments = []
        for arg in arguments[:2]:
            if isinstance(arg, str) and arg.strip():
                cleaned_arguments.append(arg.strip())
            elif arg is not None:
                cleaned_arguments.append(str(arg).strip())

        if point.strip():
            cleaned_points.append({
                "point": point.strip(),
                "arguments": cleaned_arguments,
            })

    parsed["points_of_discussion"] = cleaned_points[:3]

    if "updated_summary" not in parsed:
        parsed["updated_summary"] = ""

    if not isinstance(parsed["updated_summary"], str):
        parsed["updated_summary"] = str(parsed["updated_summary"])

    return parsed


def update_running_summary(
    llm,
    current_state: Optional[Dict[str, Any]],
    new_intervention_text: str,
    speaker: str,
    party: str,
    idx: int,
    max_words: int = 250,
) -> tuple[Dict[str, Any], str]:
    current_state_json = (
        json.dumps(current_state, ensure_ascii=False, indent=2)
        if current_state is not None
        else None
    )

    prompt = build_incremental_summary_prompt(
        current_state_json=current_state_json,
        new_intervention_text=new_intervention_text,
        speaker=speaker,
        party=party,
        idx=idx,
        max_words=max_words,
    )

    raw_output, parsed = generate_json_with_raw(
        llm=llm,
        prompt=prompt,
        max_new_tokens=700,
        temperature=0.0,
    )

    parsed = validate_state(parsed)
    return parsed, raw_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--doc_id_col", type=str, default="document_id")
    parser.add_argument("--order_col", type=str, default="intervention_id")
    parser.add_argument("--speaker_col", type=str, default="speaker")
    parser.add_argument("--party_col", type=str, default="party")
    parser.add_argument("--text_col", type=str, default="speech")

    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--add_timestamp", action="store_true", help="Add timestamp to output filenames")
    parser.add_argument("--max_words", type=int, default=250)
    add_backend_args(parser)

    return parser.parse_args()


def load_state_from_output_cell(cell_value: Any) -> Optional[Dict[str, Any]]:
    if pd.isna(cell_value):
        return None
    if not isinstance(cell_value, str) or not cell_value.strip():
        return None

    try:
        parsed = json.loads(cell_value)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None


def main():
    args = parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if args.add_timestamp else ""
    timestamp_str = f"_{timestamp}" if timestamp else ""

    df = pd.read_csv(args.input_csv)
    df = df.sort_values([args.doc_id_col, args.order_col]).reset_index(drop=True)

    llm = create_llm_from_args(args)

    # Process each document separately
    unique_docs = df[args.doc_id_col].unique()
    print(f"[DEBUG] Processing {len(unique_docs)} document(s)...", flush=True)

    for doc_id in unique_docs:
        # Generate output CSV per document
        output_csv = os.path.join(args.output_dir, f"{doc_id}_summary{timestamp_str}.csv")
        
        # Get rows for this document
        df_doc = df[df[args.doc_id_col] == doc_id].copy().reset_index(drop=True)
        
        print(f"\n[DEBUG] Processing document: {doc_id} ({len(df_doc)} rows)", flush=True)
        
        # Resume logic per document
        processed_records = []
        start_idx = 0
        running_state = None
        
        if args.resume and os.path.exists(output_csv):
            done_df = pd.read_csv(output_csv)
            start_idx = len(done_df)
            processed_records = done_df.to_dict("records")
            
            if start_idx > 0:
                running_state = load_state_from_output_cell(done_df.iloc[-1]["raw_model_output"])
            
            print(f"[DEBUG] Resume enabled for {doc_id} | start_idx={start_idx}", flush=True)
        
        # Process rows for this document
        for i in tqdm(range(start_idx, len(df_doc)), total=len(df_doc) - start_idx, desc=f"Doc {doc_id}"):
            row = df_doc.iloc[i]
            raw_model_response = ""
            
            text = str(row[args.text_col]) if pd.notna(row[args.text_col]) else ""
            word_count = len(text.split())

            summary_before = (
                running_state.get("updated_summary", "") if running_state is not None else ""
            )
            state_before_json = (
                json.dumps(running_state, ensure_ascii=False) if running_state is not None else ""
            )

            if word_count < 15:
                running_summary_after = summary_before
                raw_output = "SKIPPED: too short"
                skipped = True

            else:
                try:
                    result, raw_model_response = update_running_summary(
                        llm=llm,
                        current_state=running_state,
                        new_intervention_text=text,
                        speaker=str(row[args.speaker_col]) if pd.notna(row[args.speaker_col]) else "Onbekend",
                        party=str(row[args.party_col]) if pd.notna(row[args.party_col]) else "Onbekend",
                        idx=int(row[args.order_col]),
                        max_words=args.max_words,
                    )

                    running_state = result
                    raw_output = json.dumps(result, ensure_ascii=False)
                    running_summary_after = result.get("updated_summary", "")
                    skipped = False

                except Exception as e:
                    raw_output = f"ERROR: {str(e)}"
                    running_summary_after = summary_before
                    skipped = False

            record = row.to_dict()
            record["summary_before"] = summary_before
            record["state_before_json"] = state_before_json
            record["running_summary_after"] = running_summary_after
            record["raw_model_output"] = raw_output
            record["raw_model_response"] = raw_model_response
            record["skipped"] = skipped
            record["word_count"] = word_count

            processed_records.append(record)

            if (i + 1) % args.checkpoint_every == 0:
                out_df = pd.DataFrame(processed_records)
                safe_to_csv(out_df, output_csv)
        
        # Final save for this document
        if processed_records:
            out_df = pd.DataFrame(processed_records)
            safe_to_csv(out_df, output_csv)
            print(f"[DEBUG] Saved {len(processed_records)} rows to {output_csv}", flush=True)


if __name__ == "__main__":
    main()
