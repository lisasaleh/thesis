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
    r"\d{1,2}-\d{1,2}-\d{1,2}",   # e.g. 12-8-37
    r"\b\d+\b(?=\s*$)",           # page number at end
]

PARTIES = [
    "VVD",
    "PvdA",
    "PVV",
    "SP",
    "CDA",
    "D66",
    "GroenLinks",
    "ChristenUnie",
    "SGP",
    "PvdD",
    "50PLUS",
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
    """Clean page text by removing headers, hyphenation, and normalizing whitespace."""
    if pd.isna(text):
        return ""

    text = str(text)

    # remove hyphenation across line breaks
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # remove recurring header patterns
    for pattern in HEADER_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # remove repeated agenda intro if present
    text = re.sub(
        r"Aan de orde is .*?(?=De voorzitter:|De heer|Mevrouw|Minister|Staatssecretaris|Secretaris|$)",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_json_document(json_path: str) -> Dict:
    """Load JSON document."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_text_from_json(doc: Dict) -> str:
    """Extract and clean text from JSON document."""
    # Try multiple possible text field names
    text_candidates = [
        doc.get("foi_bodyText"),
        doc.get("dc_description"),
        doc.get("foi_text"),
    ]
    
    text = next((t for t in text_candidates if t), "")
    return clean_page_text(text) if text else ""


def split_interventions(text: str) -> List[Dict]:
    """Split text into individual interventions by speaker."""
    if not text or pd.isna(text):
        return []

    matches = list(SPEAKER_PATTERN.finditer(text))
    if not matches:
        return []

    interventions = []

    # Skip preamble (before first speaker)
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
            m = re.match(rf"(De heer .+|Mevrouw .+) \(({PARTY_PATTERN})\)$", speaker_label)
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
            "speech": speech
        })

    return interventions


def load_relevant_themes(manifest_path: str) -> Set[str]:
    """Extract all unique theme IDs from cmp_manifest.csv."""
    df = pd.read_csv(manifest_path)
    themes = set()
    
    # Extract theme IDs from all_theme_ids column
    for theme_str in df['all_theme_ids'].dropna():
        if pd.isna(theme_str) or theme_str == "":
            continue
        # Split by ;; to get individual theme IDs
        theme_ids = [t.strip() for t in str(theme_str).split(';;')]
        themes.update(theme_ids)
    
    return themes


def load_valid_documents(debates_path: str, relevant_themes: Set[str]) -> Set[str]:
    """Load document IDs that have day_count != -1 and relevant themes.
    
    Handles both single theme_id and multiple theme IDs separated by ;;
    """
    df = pd.read_csv(debates_path)
    
    # Filter by day_count != -1
    df = df[df['day_count'] != -1]
    
    # Check if theme_id contains any relevant themes
    # Handle both single themes and multiple themes separated by ;;
    def has_relevant_theme(theme_str):
        if pd.isna(theme_str):
            return False
        themes = [t.strip() for t in str(theme_str).split(';;')]
        return any(t in relevant_themes for t in themes)
    
    valid_df = df[df['theme_id'].apply(has_relevant_theme)]
    
    # Use dc_identifier (matches JSON filenames) instead of dc_externalIdentifier
    return set(valid_df['dc_identifier'].unique())


def process_document(json_path: str, document_id: str, 
                    interventions_list: List[Dict], intervention_counter: Dict) -> None:
    """Process a single JSON document and extract interventions."""
    try:
        doc = load_json_document(json_path)
        text = extract_text_from_json(doc)
        
        if not text or len(text.split()) < 10:
            return
        
        interventions = split_interventions(text)
        
        for interv in interventions:
            intervention_counter['count'] += 1
            
            interventions_list.append({
                "document_id": document_id,
                "intervention_id": intervention_counter['count'],
                "speaker_label": interv["speaker_label"],
                "speaker": interv["speaker"],
                "party": interv["party"] if interv["party"] else "",
                "speech": interv["speech"],
                "n_words": len(interv["speech"].split())
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
        help="Output directory (will be created if it doesn't exist)"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="outputs/cmp_manifest.csv",
        help="Path to cmp_manifest.csv"
    )
    parser.add_argument(
        "--debates",
        type=str,
        default="outputs/debates.csv",
        help="Path to debates.csv"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to data directory containing year folders"
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

    print("Sample JSON stems:")
    for year_folder in sorted(Path(args.data_dir).iterdir()):
        if year_folder.is_dir():
            print([p.stem for p in sorted(year_folder.glob("*.json"))[:10]])
            break
    
    # Process documents by year
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    total_interventions = 0
    
    # Iterate through years in data directory
    for year_folder in sorted(data_dir.iterdir()):
        if not year_folder.is_dir():
            continue
        
        year = year_folder.name
        year_output_dir = output_dir / year
        year_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing year {year}...")
        
        # Find all JSON files in this year
        json_files = sorted(year_folder.glob("*.json"))
        
        if not json_files:
            print(f"  No JSON files found in {year_folder}")
            continue
        
        year_interventions = 0
        year_processed = 0
        
        for json_path in json_files:
            # Extract document ID (filename without extension)
            document_id = json_path.stem
            
            # Check if this document is valid
            if document_id not in valid_docs:
                continue
            
            interventions_list = []
            intervention_counter = {'count': 0}
            
            process_document(str(json_path), document_id, interventions_list, intervention_counter)
            
            if interventions_list:
                # Save to CSV
                df = pd.DataFrame(interventions_list)
                output_path = year_output_dir / f"{document_id}.csv"
                df.to_csv(output_path, index=False)
                
                year_processed += 1
                year_interventions += len(interventions_list)
                total_interventions += len(interventions_list)
                processed_count += 1
        
        print(f"  Processed {year_processed} documents with {year_interventions} total interventions")
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Total documents processed: {processed_count}")
    print(f"Total interventions extracted: {total_interventions}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
