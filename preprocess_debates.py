import re
import json
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Set


HEADER_PATTERNS = [
    r"Tweede Kamer",
    r"Eerste Kamer",
    r"\bTK\b\s+\d+",
    r"\bAH\b\s+\d+",
    r"\d{1,2}-\d{1,2}-\d{1,2}",
    r"\b\d+\b(?=\s*$)",
]

PARTIES = [
    "VVD", "PvdA", "PVV", "SP", "CDA", "D66", "GroenLinks",
    "ChristenUnie", "SGP", "PvdD", "50PLUS",
]

PARTY_PATTERN = "|".join(re.escape(p) for p in PARTIES)

SPEAKER_PATTERN = re.compile(
    rf"(De voorzitter|"
    rf"De heer [A-Za-zÀ-ÿ'`\- ]+ \(({PARTY_PATTERN})\)|"
    rf"Mevrouw [A-Za-zÀ-ÿ'`\- ]+ \(({PARTY_PATTERN})\)|"
    rf"Minister [A-Za-zÀ-ÿ'`\- ]+|"
    rf"Staatssecretaris [A-Za-zÀ-ÿ'`\- ]+):"
)


def clean_page_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)

    for pattern in HEADER_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(
        r"Aan de orde is .*?(?=De voorzitter:|De heer|Mevrouw|Minister|Staatssecretaris|Secretaris|$)",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    return re.sub(r"\s+", " ", text).strip()


def load_json_document(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text_from_json(doc: Dict) -> str:
    """Extract and clean text from JSON document by joining all pages."""
    all_text = []
    
    # Extract text from all pages in foi_files
    foi_files = doc.get("foi_files", [])
    
    for file_obj in foi_files:
        foi_pages = file_obj.get("foi_pages", [])
        
        # Sort pages by page number to maintain order
        foi_pages_sorted = sorted(foi_pages, key=lambda p: p.get("foi_pageNumber", 0))
        
        for page in foi_pages_sorted:
            page_text = page.get("foi_bodyText")
            if page_text:
                all_text.append(clean_page_text(page_text))
    
    # Join all pages with space
    full_text = " ".join(all_text)
    return re.sub(r"\s+", " ", full_text).strip()


def split_interventions(text: str) -> List[Dict]:
    if not text or pd.isna(text):
        return []

    matches = list(SPEAKER_PATTERN.finditer(text))
    if not matches:
        return []

    interventions = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        chunk = text[start:end].strip()
        colon_idx = chunk.find(":")

        if colon_idx == -1:
            continue

        speaker_label = chunk[:colon_idx].strip()
        speech = chunk[colon_idx + 1:].strip()

        if speaker_label == "De voorzitter":
            speaker = "De voorzitter"
            party = None
        else:
            m = re.match(
                rf"(De heer .+|Mevrouw .+) \(({PARTY_PATTERN})\)$",
                speaker_label,
            )
            if m:
                speaker = m.group(1).strip()
                party = m.group(2).strip()
            else:
                speaker = speaker_label
                party = None

        if len(speech.split()) == 0:
            continue

        interventions.append({
            "speaker_label": speaker_label,
            "speaker": speaker,
            "party": party,
            "speech": speech,
        })

    return interventions


def load_relevant_themes(manifest_path: str) -> Set[str]:
    df = pd.read_csv(manifest_path)

    themes = set()

    for theme_str in df["all_theme_ids"].dropna():
        theme_ids = [t.strip() for t in str(theme_str).split(";;")]
        themes.update(theme_ids)

    return themes


def load_valid_documents(debates_path: str, relevant_themes: Set[str]) -> Set[str]:
    df = pd.read_csv(debates_path)

    df = df[df["day_count"] != -1]

    def has_relevant_theme(theme_str):
        if pd.isna(theme_str):
            return False

        themes = [t.strip() for t in str(theme_str).split(";;")]
        return any(t in relevant_themes for t in themes)

    valid_df = df[df["theme_id"].apply(has_relevant_theme)]

    return set(
        valid_df["dc_identifier"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )


def process_document(
    json_path: str,
    document_id: str,
    interventions_list: List[Dict],
    intervention_counter: Dict,
    debug: bool = False,
) -> None:
    try:
        doc = load_json_document(json_path)
        text = extract_text_from_json(doc)

        if debug:
            print(f"\nDEBUG document: {document_id}")
            print("JSON keys:", list(doc.keys()))
            print("Extracted text words:", len(text.split()))
            print("Extracted text sample:")
            print(text[:1000])

        if not text or len(text.split()) < 10:
            return

        interventions = split_interventions(text)

        if debug:
            print("Speaker matches:", len(list(SPEAKER_PATTERN.finditer(text))))
            print("Interventions extracted:", len(interventions))

        for interv in interventions:
            intervention_counter["count"] += 1

            interventions_list.append({
                "document_id": document_id,
                "intervention_id": intervention_counter["count"],
                "speaker_label": interv["speaker_label"],
                "speaker": interv["speaker"],
                "party": interv["party"] if interv["party"] else "",
                "speech": interv["speech"],
                "n_words": len(interv["speech"].split()),
            })

    except Exception as e:
        print(f"Error processing {json_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess debates into intervention-level CSVs"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch-shared/lsaleh/debates",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="outputs/cmp_manifest.csv",
    )
    parser.add_argument(
        "--debates",
        type=str,
        default="outputs/debates.csv",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information for the first matching document.",
    )

    args = parser.parse_args()

    print("Loading relevant themes from manifest...")
    relevant_themes = load_relevant_themes(args.manifest)
    print(f"Found {len(relevant_themes)} unique themes in manifest")

    print("Loading valid documents from debates.csv...")
    valid_docs = load_valid_documents(args.debates, relevant_themes)
    print(f"Found {len(valid_docs)} valid documents (day_count != -1, relevant themes)")

    print("Sample valid_docs:")
    print(list(valid_docs)[:10])

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    total_interventions = 0
    debug_done = False

    for year_folder in sorted(data_dir.iterdir()):
        if not year_folder.is_dir():
            continue

        year = year_folder.name
        year_output_dir = output_dir / year
        year_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing year {year}...")

        json_files = sorted(year_folder.glob("*.json"))

        if not json_files:
            print(f"  No JSON files found in {year_folder}")
            continue

        json_stems = {p.stem.strip() for p in json_files}
        matches = json_stems & valid_docs

        print(f"  JSON files: {len(json_files)}")
        print(f"  Matching valid docs: {len(matches)}")
        print(f"  Example matches: {list(matches)[:5]}")

        year_interventions = 0
        year_processed = 0

        for json_path in json_files:
            document_id = json_path.stem.strip()

            if document_id not in valid_docs:
                continue

            interventions_list = []
            intervention_counter = {"count": 0}

            run_debug = args.debug and not debug_done

            process_document(
                str(json_path),
                document_id,
                interventions_list,
                intervention_counter,
                debug=run_debug,
            )

            if run_debug:
                debug_done = True

            if interventions_list:
                df = pd.DataFrame(interventions_list)
                output_path = year_output_dir / f"{document_id}.csv"
                df.to_csv(output_path, index=False)

                year_processed += 1
                year_interventions += len(interventions_list)
                total_interventions += len(interventions_list)
                processed_count += 1

        print(
            f"  Processed {year_processed} documents "
            f"with {year_interventions} total interventions"
        )

    print(f"\n{'=' * 50}")
    print("Processing complete!")
    print(f"Total documents processed: {processed_count}")
    print(f"Total interventions extracted: {total_interventions}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()