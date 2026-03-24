import argparse
import json
import os
from typing import List

import pandas as pd
from tqdm import tqdm

from llm_utils import LocalLLM, extract_json


def build_incremental_summary_prompt(
    current_summary: str,
    speaker: str,
    party: str,
    idx: int,
    new_intervention_text: str,
    max_words: int = 200,
) -> str:
    return f"""
Je helpt bij het incrementeel samenvatten van een parlementair debat.

Doel:
Werk de lopende samenvatting van het debat bij op basis van de NIEUWE INTERVENTIE, zodat latere interventies goed in hun politieke en argumentatieve context kunnen worden geïnterpreteerd.

Taak:
Verwerk de inhoud van de NIEUWE INTERVENTIE in de bestaande samenvatting van het debat tot nu toe.

Instructies:
- Schrijf alles in het Nederlands.
- Geef alleen geldige JSON terug.
- Focus uitsluitend op inhoudelijke politieke inhoud.
- Behoud het hoofdonderwerp van het debat, de standpunten van sprekers en partijen, meningsverschillen, argumenten, voorstellen, bezwaren en relevante reacties.
- Voeg nieuwe relevante informatie uit de NIEUWE INTERVENTIE toe aan de samenvatting.
- Laat begroetingen, procedurele opmerkingen, humor en retorische opvulling weg, tenzij ze inhoudelijk relevant zijn voor het politieke standpunt of de argumentatie.
- Zorg dat de samenvatting cumulatief blijft: belangrijke eerdere informatie mag niet verdwijnen tenzij zij duidelijk irrelevant is geworden.
- Vat conflicten en verschillen in positie expliciet samen; maak tegengestelde standpunten niet onnodig vaag of algemeen.
- Geef de voorkeur aan informatiedichtheid en duidelijkheid boven mooie formulering.
- Verzin geen informatie en trek geen conclusies die niet in de tekst staan.
- Houd de bijgewerkte samenvatting onder de {max_words} woorden.

JSON-schema:
{{
  "updated_summary": "",
  "new_information_added": [
    ""
  ]
}}

HUIDIGE LOPENDE SAMENVATTING:
{current_summary if current_summary else "Nog geen samenvatting."}

METADATA VAN DE NIEUWE INTERVENTIE:
Spreker: {speaker}
Partij: {party}
Interventie-index: {idx}

NIEUWE INTERVENTIE:
{new_intervention_text}
""".strip()


def update_running_summary(
    llm: LocalLLM,
    current_summary: str,
    new_intervention_text: str,
    speaker: str,
    party: str,
    idx: int,
) -> dict:
    prompt = build_incremental_summary_prompt(
        current_summary=current_summary,
        new_intervention_text=new_intervention_text,
        speaker=speaker,
        party=party,
        idx=idx,
    )

    raw_output = llm.generate(
        prompt=prompt,
        max_new_tokens=300,
        temperature=0.2,
    )

    parsed = extract_json(raw_output)

    if "updated_summary" not in parsed:
        raise ValueError(f"Missing 'updated_summary' in output: {parsed}")

    if "new_information_added" not in parsed:
        parsed["new_information_added"] = []

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
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = df.sort_values([args.doc_id_col, args.order_col]).reset_index(drop=True)

    llm = LocalLLM(args.model_name)

    if args.resume and os.path.exists(args.output_csv):
        done_df = pd.read_csv(args.output_csv)
        start_idx = len(done_df)

        if start_idx > 0:
            current_doc_id = done_df.iloc[-1][args.doc_id_col]
            running_summary = done_df.iloc[-1]["running_summary_after"]
        else:
            start_idx = 0
            current_doc_id = None
            running_summary = ""
    else:
        start_idx = 0
        current_doc_id = None
        running_summary = ""

    records = []

    for i in tqdm(range(start_idx, len(df)), total=len(df) - start_idx):
        row = df.iloc[i]
        row_doc_id = row[args.doc_id_col]

        if row_doc_id != current_doc_id:
            current_doc_id = row_doc_id
            running_summary = ""

        summary_before = running_summary

        try:
            result = update_running_summary(
                llm=llm,
                current_summary=running_summary,
                new_intervention_text=str(row[args.text_col]) if pd.notna(row[args.text_col]) else "",
                speaker=str(row[args.speaker_col]) if pd.notna(row[args.speaker_col]) else "Unknown",
                party=str(row[args.party_col]) if pd.notna(row[args.party_col]) else "Unknown",
                idx=int(row[args.order_col]),
            )
            running_summary = result["updated_summary"]
            new_info = json.dumps(result.get("new_information_added", []), ensure_ascii=False)
            raw_output = json.dumps(result, ensure_ascii=False)

        except Exception as e:
            new_info = f"ERROR: {str(e)}"
            raw_output = f"ERROR: {str(e)}"
            # Keep previous running summary on failure

        record = row.to_dict()
        record["summary_before"] = summary_before
        record["summary_update_info"] = new_info
        record["running_summary_after"] = running_summary
        record["raw_model_output"] = raw_output
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

    print(f"Saved output to {args.output_csv}")


if __name__ == "__main__":
    main()