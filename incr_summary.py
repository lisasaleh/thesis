import argparse
import json
import os
from typing import Optional, Dict, Any

import pandas as pd
from tqdm import tqdm

from llm_utils import LocalLLM, generate_json
from prompts.summary_prompt import build_incremental_summary_prompt


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
    llm: LocalLLM,
    current_state: Optional[Dict[str, Any]],
    new_intervention_text: str,
    speaker: str,
    party: str,
    idx: int,
    max_words: int = 250,
) -> Dict[str, Any]:
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

    parsed = generate_json(
        llm=llm,
        prompt=prompt,
        max_new_tokens=700,
        temperature=0.0,
    )

    parsed = validate_state(parsed)
    return parsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--doc_id_col", type=str, default="document_id")
    parser.add_argument("--order_col", type=str, default="intervention_id")
    parser.add_argument("--speaker_col", type=str, default="speaker")
    parser.add_argument("--party_col", type=str, default="party")
    parser.add_argument("--text_col", type=str, default="speech")

    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_words", type=int, default=250)

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

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = df.sort_values([args.doc_id_col, args.order_col]).reset_index(drop=True)

    print("[DEBUG] Starting model load...", flush=True)
    llm = LocalLLM(args.model_name)
    print("[DEBUG] Model load finished.", flush=True)

    if args.resume and os.path.exists(args.output_csv):
        done_df = pd.read_csv(args.output_csv)
        start_idx = len(done_df)

        if start_idx > 0:
            current_doc_id = done_df.iloc[-1][args.doc_id_col]
            running_state = load_state_from_output_cell(done_df.iloc[-1]["raw_model_output"])
        else:
            start_idx = 0
            current_doc_id = None
            running_state = None
    else:
        start_idx = 0
        current_doc_id = None
        running_state = None

    records = []

    print(f"[DEBUG] Starting from index {start_idx}", flush=True)

    for i in tqdm(range(start_idx, len(df)), total=len(df) - start_idx):
        row = df.iloc[i]
        row_doc_id = row[args.doc_id_col]

        if row_doc_id != current_doc_id:
            current_doc_id = row_doc_id
            running_state = None

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
                result = update_running_summary(
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
                # running_state blijft ongewijzigd

        record = row.to_dict()
        record["summary_before"] = summary_before
        record["state_before_json"] = state_before_json
        record["running_summary_after"] = running_summary_after
        record["raw_model_output"] = raw_output
        record["skipped"] = skipped
        record["word_count"] = word_count

        records.append(record)

        if (i + 1) % args.checkpoint_every == 0:
            if args.resume and os.path.exists(args.output_csv):
                prev_df = pd.read_csv(args.output_csv)
                out_df = pd.concat([prev_df, pd.DataFrame(records)], ignore_index=True)
            else:
                prefix_df = df.iloc[:start_idx].copy() if start_idx > 0 else pd.DataFrame()
                out_df = pd.concat([prefix_df, pd.DataFrame(records)], ignore_index=True)

            out_df.to_csv(args.output_csv, index=False)
            records = []

    if records:
        if args.resume and os.path.exists(args.output_csv):
            prev_df = pd.read_csv(args.output_csv)
            out_df = pd.concat([prev_df, pd.DataFrame(records)], ignore_index=True)
        else:
            prefix_df = df.iloc[:start_idx].copy() if start_idx > 0 else pd.DataFrame()
            out_df = pd.concat([prefix_df, pd.DataFrame(records)], ignore_index=True)

        out_df.to_csv(args.output_csv, index=False)

    print(f"[DEBUG] Saved output to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()