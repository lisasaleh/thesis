import re
import json
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Set, Optional


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
NAME_CHARS = r"A-Za-zÀ-ÿ'`’\-. ]+"

# After text normalization, these labels should contain normal spaces.
SPEAKER_PATTERN = re.compile(
    rf"(De voorzitter|"
    rf"De heer {NAME_CHARS} \(({PARTY_PATTERN})\)|"
    rf"Mevrouw {NAME_CHARS} \(({PARTY_PATTERN})\)|"
    rf"Minister {NAME_CHARS}|"
    rf"Staatssecretaris {NAME_CHARS}):"
)


def normalize_pdf_artifacts(text: str) -> str:
    """Normalize common PDF/OCR extraction artifacts before regex parsing.

    The official-doc PDFs often contain U+FFFF-like separator artifacts, e.g.
    'Mevrouw\uffffHelder\uffff(PVV):'. Without replacing these by spaces,
    speaker detection fails and valid debates are marked as unprocessed.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    replacements = {
        "\uffff": " ",
        "\ufffe": " ",
        "\ufffd": " ",
        "\u00a0": " ",
        "\u00ad": "",   # soft hyphen
        "\u2028": " ",
        "\u2029": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Replace any remaining specials in the Unicode Specials block by spaces.
    text = re.sub(r"[\uFFF0-\uFFFF]", " ", text)
    return text


def clean_page_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = normalize_pdf_artifacts(text)

    # Remove hyphenation across line/page extraction boundaries.
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    # Normalize whitespace before applying header/agenda patterns.
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


def extract_text_from_json(doc: Dict, prefer_ocr: bool = False) -> str:
    all_text = []

    for file_obj in doc.get("foi_files", []):
        foi_pages = file_obj.get("foi_pages", [])
        foi_pages_sorted = sorted(
            foi_pages,
            key=lambda p: p.get("foi_pageNumber", 0),
        )

        for page in foi_pages_sorted:
            if prefer_ocr:
                page_text = page.get("foi_bodyTextOCR") or page.get("foi_bodyText")
            else:
                page_text = page.get("foi_bodyText") or page.get("foi_bodyTextOCR")
            if page_text:
                all_text.append(clean_page_text(page_text))

    full_text = " ".join(all_text)
    return re.sub(r"\s+", " ", full_text).strip()


def split_interventions(text: str) -> List[Dict]:
    if not text or pd.isna(text):
        return []

    text = clean_page_text(text)
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

        speaker_label = re.sub(r"\s+", " ", chunk[:colon_idx]).strip()
        speech = re.sub(r"\s+", " ", chunk[colon_idx + 1:]).strip()

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
        themes.update(t for t in theme_ids if t)

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


def load_doc_list(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    docs = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("valid documents"):
                continue
            docs.add(line)
    return docs


def process_document(
    json_path: str,
    document_id: str,
    interventions_list: List[Dict],
    intervention_counter: Dict,
    debug: bool = False,
    prefer_ocr: bool = False,
) -> Dict:
    result = {
        "status": "ok",
        "reason": "",
        "n_words": 0,
        "speaker_matches": 0,
        "n_interventions": 0,
    }

    try:
        doc = load_json_document(json_path)
        text = extract_text_from_json(doc, prefer_ocr=prefer_ocr)

        result["n_words"] = len(text.split())

        if debug:
            print(f"\nDEBUG document: {document_id}")
            print("JSON keys:", list(doc.keys()))
            print("Extracted text words:", result["n_words"])
            print("Extracted text sample:")
            print(text[:1000])

        if not text or result["n_words"] < 10:
            result["status"] = "skipped"
            result["reason"] = "empty_or_short_text"
            return result

        speaker_matches = list(SPEAKER_PATTERN.finditer(text))
        result["speaker_matches"] = len(speaker_matches)

        interventions = split_interventions(text)
        result["n_interventions"] = len(interventions)

        if debug:
            print("Speaker matches:", result["speaker_matches"])
            print("Interventions extracted:", result["n_interventions"])
            if interventions:
                print("First intervention:", interventions[0])

        if not interventions:
            result["status"] = "skipped"
            result["reason"] = "no_interventions_extracted"
            return result

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

        return result

    except Exception as e:
        result["status"] = "error"
        result["reason"] = str(e)
        print(f"Error processing {json_path}: {e}")
        return result


def write_doc_list(path: Path, docs: Set[str], title: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}: {len(docs)} documents\n\n")
        for doc_id in sorted(docs):
            f.write(f"{doc_id}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess debates into intervention-level CSVs, with recovery for PDF separator artifacts."
    )

    parser.add_argument("--output_dir", type=str, default="/scratch-shared/lsaleh/debates")
    parser.add_argument("--manifest", type=str, default="outputs/cmp_manifest.csv")
    parser.add_argument("--debates", type=str, default="outputs/debates.csv")
    parser.add_argument("--data_dir", type=str, default="data/raw_debates")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--save_csvs",
        action="store_true",
        help="Actually save intervention CSVs. By default, only diagnostics are written.",
    )
    parser.add_argument(
        "--only_unprocessed_list",
        type=str,
        default="/scratch-shared/lsaleh/debates/unprocessed_documents.txt",
        help="Optional path to unprocessed_documents.txt. If provided, only these document IDs are processed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-document CSVs. By default, existing CSVs are skipped.",
    )
    parser.add_argument(
        "--prefer_ocr",
        action="store_true",
        help="Prefer foi_bodyTextOCR over foi_bodyText when both are present.",
    )

    args = parser.parse_args()

    print("Loading relevant themes from manifest...")
    relevant_themes = load_relevant_themes(args.manifest)
    print(f"Found {len(relevant_themes)} unique themes in manifest")

    print("Loading valid documents from debates.csv...")
    valid_docs = load_valid_documents(args.debates, relevant_themes)
    print(f"Found {len(valid_docs)} valid documents (day_count != -1, relevant themes)")

    requested_docs = load_doc_list(args.only_unprocessed_list)
    if requested_docs:
        valid_docs = valid_docs & requested_docs
        print(f"Restricting to unprocessed list: {len(valid_docs)} relevant requested documents")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    processed_count = 0
    total_interventions = 0
    debug_done = False

    found_docs = set()
    processed_docs = set()
    already_existing_docs = set()
    unprocessed_docs = set()
    error_docs = set()

    skip_records = []

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
        print(f"  Matching valid/requested docs: {len(matches)}")
        print(f"  Example matches: {list(matches)[:5]}")

        year_interventions = 0
        year_processed = 0
        year_existing = 0

        for json_path in json_files:
            document_id = json_path.stem.strip()

            if document_id not in valid_docs:
                continue

            found_docs.add(document_id)
            output_path = year_output_dir / f"{document_id}.csv"

            if output_path.exists() and not args.overwrite:
                already_existing_docs.add(document_id)
                year_existing += 1
                continue

            interventions_list = []
            intervention_counter = {"count": 0}
            run_debug = args.debug and not debug_done

            result = process_document(
                str(json_path),
                document_id,
                interventions_list,
                intervention_counter,
                debug=run_debug,
                prefer_ocr=args.prefer_ocr,
            )

            if run_debug:
                debug_done = True

            if interventions_list:
                processed_docs.add(document_id)

                if args.save_csvs:
                    df = pd.DataFrame(interventions_list)
                    df.to_csv(output_path, index=False)

                year_processed += 1
                year_interventions += len(interventions_list)
                total_interventions += len(interventions_list)
                processed_count += 1
            else:
                unprocessed_docs.add(document_id)

                if result["status"] == "error":
                    error_docs.add(document_id)

                skip_records.append({
                    "document_id": document_id,
                    "year": year,
                    "reason": result["reason"],
                    "n_words": result["n_words"],
                    "speaker_matches": result["speaker_matches"],
                    "n_interventions": result["n_interventions"],
                })

        print(
            f"  Newly processed {year_processed} documents "
            f"with {year_interventions} total interventions"
        )
        if year_existing:
            print(f"  Skipped {year_existing} documents because CSV already exists")

    missing_docs = valid_docs - found_docs

    print(f"\n{'=' * 50}")
    print("Processing complete!")
    print(f"Valid/requested documents: {len(valid_docs)}")
    print(f"Found in data folders: {len(found_docs)}")
    print(f"Already had CSVs: {len(already_existing_docs)}")
    print(f"Newly processed with interventions: {len(processed_docs)}")
    print(f"Found but no interventions extracted: {len(unprocessed_docs)}")
    print(f"Not found in data folders: {len(missing_docs)}")
    print(f"Errors: {len(error_docs)}")
    print(f"Total interventions extracted in this run: {total_interventions}")
    print(f"Output directory: {output_dir}")
    print(f"CSV saving enabled: {args.save_csvs}")

    suffix = "recovery" if requested_docs else ""
    name = lambda base: f"{base}_{suffix}.txt" if suffix else f"{base}.txt"

    write_doc_list(
        output_dir / name("missing_documents"),
        missing_docs,
        "Valid documents not found in data folders",
    )

    write_doc_list(
        output_dir / name("unprocessed_documents"),
        unprocessed_docs,
        "Valid documents found but no interventions extracted",
    )

    write_doc_list(
        output_dir / name("processed_documents"),
        processed_docs | already_existing_docs,
        "Valid documents processed with interventions or already existing",
    )

    if skip_records:
        skip_df = pd.DataFrame(skip_records)
        skip_path = output_dir / ("unprocessed_documents_diagnostics_recovery.csv" if requested_docs else "unprocessed_documents_diagnostics.csv")
        skip_df.to_csv(skip_path, index=False)
        print(f"Unprocessed diagnostics saved to: {skip_path}")

    print(f"Missing document list saved to: {output_dir / name('missing_documents')}")
    print(f"Unprocessed document list saved to: {output_dir / name('unprocessed_documents')}")
    print(f"Processed document list saved to: {output_dir / name('processed_documents')}")


if __name__ == "__main__":
    main()
