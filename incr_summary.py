import argparse
import json
import os
from typing import Optional, Dict, Any

import pandas as pd
from tqdm import tqdm

from llm_utils import LocalLLM, extract_json_with_repair


def build_incremental_summary_prompt(
    current_state_json: Optional[str],
    speaker: str,
    party: str,
    idx: int,
    new_intervention_text: str,
    max_words: int = 400,
) -> str:
    return f"""
Je helpt bij het incrementeel bijhouden van een gestructureerde samenvatting van een Nederlands parlementair debat.

Doel:
Werk de lopende debatrepresentatie bij op basis van de NIEUWE INTERVENTIE, zodat de output steeds een samenvatting blijft van het GEHELE debat tot nu toe, niet alleen van de meest recente uitwisseling.

BELANGRIJK:
- Schrijf alle tekst volledig in het Nederlands.
- Verzin geen informatie.
- Behoud belangrijke eerdere context.
- De output moet ALLE relevante eerdere discussiepunten blijven bevatten.
- De samenvatting moet het HELE debat tot nu toe representeren.
- Het is NIET toegestaan om alleen de laatste interventie samen te vatten.
- Verwijder eerdere discussiepunten alleen als zij duidelijk niet langer relevant zijn.
- Als de nieuwe interventie voortbouwt op een bestaand discussiepunt, werk dat punt dan bij in plaats van de samenvatting te vernauwen tot alleen de laatste bijdrage.

BELANGRIJK VOOR JSON:
- Geef EXACT één JSON-object terug.
- Gebruik GEEN trailing commas.
- Gebruik GEEN extra tekst buiten JSON.
- Gebruik standaard JSON-notatie.
- Gebruik dubbele aanhalingstekens zoals gebruikelijk in JSON.

Instructies:
- Focus op inhoudelijke politieke inhoud.
- Negeer begroetingen, procedurele opmerkingen, humor en retorische opvulling, tenzij inhoudelijk relevant.
- Vat het debat samen als een verzameling terugkerende discussiepunten.
- Noteer per discussiepunt de belangrijkste argumenten, bezwaren, voorstellen of reacties die tot nu toe zijn genoemd.
- Zorg dat de uiteindelijke "updated_summary" een compacte samenvatting is van het gehele debat tot nu toe.
- Gebruik maximaal 4 discussiepunten.
- Gebruik maximaal 3 argumenten per discussiepunt.
- Combineer vergelijkbare argumenten.
- Houd "updated_summary" onder de {max_words} woorden.

JSON-schema:
{{
  "main_topic": "",
  "points_of_discussion": [
    {{
      "point": "",
      "arguments": [
        ""
      ]
    }}
  ],
  "updated_summary": "",
  "new_information_added": [
    ""
  ]
}}

Voorbeeld van correcte uitvoer:
{{
  "main_topic": "Intrekking van het Nederlanderschap bij terroristische misdrijven",
  "points_of_discussion": [
    {{
      "point": "Proportionaliteit van de maatregel",
      "arguments": [
        "De minister stelt dat proportionaliteit gewaarborgd is door rechterlijke toetsing.",
        "Critici vrezen dat de maatregel te ver gaat en mogelijk leidt tot stateloosheid."
      ]
    }}
  ],
  "updated_summary": "Het debat gaat over de proportionaliteit en rechtsstatelijke legitimiteit van het intrekken van het Nederlanderschap bij terroristische misdrijven.",
  "new_information_added": [
    "De minister benadrukt dat het gaat om terroristische misdrijven en geen symptoombestrijding."
  ]
}}

HUIDIGE LOPENDE DEBATSTAAT:
{current_state_json if current_state_json else "Nog geen samenvatting."}

METADATA VAN DE NIEUWE INTERVENTIE:
Spreker: {speaker}
Partij: {party}
Interventie-index: {idx}

NIEUWE INTERVENTIE:
{new_intervention_text}
""".strip()


def validate_state(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if "main_topic" not in parsed:
        parsed["main_topic"] = ""

    if "points_of_discussion" not in parsed or not isinstance(parsed["points_of_discussion"], list):
        parsed["points_of_discussion"] = []

    if "updated_summary" not in parsed:
        raise ValueError(f"Missing 'updated_summary' in output: {parsed}")

    if "new_information_added" not in parsed or not isinstance(parsed["new_information_added"], list):
        parsed["new_information_added"] = []

    return parsed


def update_running_summary(
    llm: LocalLLM,
    current_state: Optional[Dict[str, Any]],
    new_intervention_text: str,
    speaker: str,
    party: str,
    idx: int,
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
    )

    raw_output = llm.generate(
        prompt=prompt,
        max_new_tokens=400,
        temperature=0.0,
    )

    parsed = extract_json_with_repair(raw_output, llm=llm)    
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

    llm = LocalLLM(args.model_name)

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

    for i in tqdm(range(start_idx, len(df)), total=len(df) - start_idx):
        row = df.iloc[i]
        row_doc_id = row[args.doc_id_col]

        if row_doc_id != current_doc_id:
            current_doc_id = row_doc_id
            running_state = None

        summary_before = (
            running_state.get("updated_summary", "") if running_state is not None else ""
        )
        state_before_json = (
            json.dumps(running_state, ensure_ascii=False) if running_state is not None else ""
        )

        try:
            result = update_running_summary(
                llm=llm,
                current_state=running_state,
                new_intervention_text=str(row[args.text_col]) if pd.notna(row[args.text_col]) else "",
                speaker=str(row[args.speaker_col]) if pd.notna(row[args.speaker_col]) else "Onbekend",
                party=str(row[args.party_col]) if pd.notna(row[args.party_col]) else "Onbekend",
                idx=int(row[args.order_col]),
            )

            running_state = result
            new_info = json.dumps(result.get("new_information_added", []), ensure_ascii=False)
            raw_output = json.dumps(result, ensure_ascii=False)
            running_summary_after = result.get("updated_summary", "")

        except Exception as e:
            new_info = f"ERROR: {str(e)}"
            raw_output = f"ERROR: {str(e)}"
            running_summary_after = summary_before
            # keep previous running_state on failure

        record = row.to_dict()
        record["summary_before"] = summary_before
        record["state_before_json"] = state_before_json
        record["summary_update_info"] = new_info
        record["running_summary_after"] = running_summary_after
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