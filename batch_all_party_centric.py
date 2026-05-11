"""
Party-Centric Batch Pipeline: extract → summarize (unbiased) → normalize per (party, debate)

Architecture:
1. Main argument: --party (one party per run)
2. Load cmp_manifest to get top 5 CMP codes for that party (indexed 1-5)
3. For each CMP code rank (1-5):
   - Get theme_ids from that CMP code
   - Find debates in outputs/debates.csv that match those themes
   - For each matching debate:
     - Load split intervention CSV (e.g., nl.oorg10002.2b.2012.20122013-30-19.csv)
     - Aggregate text spoken by target party
     - Check if >= min_tokens threshold
     - If yes: run extract → summarize (unbiased) → normalize pipeline
    - Save as /scratch-shared/lsaleh/debates/{stage}/{party}_{debate_id}_*.csv
    - Reuse summaries across all CMP codes for the same debate
4. Resumability: skip if output already exists, checkpoint stores debate_id
5. Logging: track below-threshold debates, empty extractions, processing status

Usage:
    python batch_all_party_centric.py --party VVD --model_7b Qwen/Qwen2.5-7B-Instruct \\
        --model_32b /path/to/Qwen2.5-32B-Instruct --min_tokens 100
"""

import os
import subprocess
import argparse
import json
import glob
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
from tqdm import tqdm


# Party name aliases (normalize inconsistent party names in debates CSV vs cmp_manifest)
PARTY_ALIASES = {
    "PVDA": "PvdA",
    "PVDD": "PvdD",
    "GL": "GroenLinks",
    "CU": "ChristenUnie",
}


# ============================================================================
# Logging & Checkpointing
# ============================================================================

class ProcessingLogger:
    """Track processing status and skip reasons."""
    
    def __init__(self, party: str, log_dir: str = "outputs/logs"):
        self.party = party
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(
            log_dir, 
            f"{party}_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.checkpoint_file = os.path.join(log_dir, f".checkpoint_{party}.json")
        
    def log(self, message: str, level: str = "INFO"):
        """Write log message to file and print."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] [{level}] {message}"
        print(full_msg)
        with open(self.log_file, 'a') as f:
            f.write(full_msg + "\n")
    
    def log_skipped(self, cmp_rank: int, cmp_code: int, debate_id: str, reason: str):
        """Log a skipped case."""
        msg = f"SKIPPED: cmp_rank={cmp_rank}, cmp_code={cmp_code}, debate_id={debate_id}, reason={reason}"
        self.log(msg, level="SKIP")
    
    def save_checkpoint(self, completed: Set[str]):
        """Save completed debate_ids."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                "completed": sorted(list(completed))
            }, f, indent=2)
    
    def load_checkpoint(self) -> Set[str]:
        """Load completed debate_ids."""
        if not os.path.exists(self.checkpoint_file):
            return set()
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                return set(item for item in data.get("completed", []))
        except:
            return set()


# ============================================================================
# Data Loading & Theme Matching
# ============================================================================

def load_cmp_manifest(csv_path: str = "/scratch-shared/lsaleh/debates/cmp_manifest.csv") -> pd.DataFrame:
    """Load CMP manifest with party → top 5 CMP codes mapping."""
    return pd.read_csv(csv_path)


def normalize_party_name(party: str) -> str:
    """Normalize party name using aliases."""
    return PARTY_ALIASES.get(party, party)


def get_debate_date(debate_row: pd.Series) -> str:
    """Return a stable YYYY-MM-DD date string for a debate row."""
    for column in ("foi_meetingDate", "dc_date", "date"):
        if column in debate_row and pd.notna(debate_row[column]):
            return str(debate_row[column]).strip()
    return ""


def build_debate_metadata_rows(
    party: str,
    cmp_info: Dict,
    debate_ids: List[str],
    debates_csv: str
) -> List[Dict[str, object]]:
    """Build one metadata row per party x cmp_rank x debate_id."""
    if not debate_ids:
        return []

    debates_df = pd.read_csv(debates_csv)
    debate_lookup = debates_df.drop_duplicates(subset=["dc_identifier"]).set_index("dc_identifier")

    rows: List[Dict[str, object]] = []
    date_counts = {}

    for debate_id in debate_ids:
        if debate_id in debate_lookup.index:
            date_value = get_debate_date(debate_lookup.loc[debate_id])
        else:
            date_value = ""
        date_counts[date_value] = date_counts.get(date_value, 0) + 1
        rows.append({
            "party": party,
            "cmp_rank": cmp_info["rank"],
            "cmp_code": cmp_info["code"],
            "cmp_title": cmp_info.get("title", ""),
            "debate_id": debate_id,
            "date": date_value,
        })

    for row in rows:
        row["datecount"] = date_counts.get(row["date"], 0)
        row["debate_count_for_combo"] = len(debate_ids)

    return rows


def write_metadata_csv(metadata_rows: List[Dict[str, object]], metadata_csv: str):
    """Append metadata rows to a CSV in outputs/, deduplicating exact rows."""
    if not metadata_rows:
        return

    new_df = pd.DataFrame(metadata_rows)
    if os.path.exists(metadata_csv):
        existing_df = pd.read_csv(metadata_csv)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["party", "cmp_rank", "cmp_code", "debate_id"], keep="last")
    else:
        combined = new_df

    combined = combined.sort_values(["party", "cmp_rank", "cmp_code", "date", "debate_id"])
    os.makedirs(os.path.dirname(metadata_csv), exist_ok=True)
    combined.to_csv(metadata_csv, index=False)


def get_party_cmp_codes(cmp_manifest: pd.DataFrame, party: str) -> List[Dict]:
    """
    Get top 5 CMP codes for a party with their theme_ids.
    
    Returns list of dicts:
        [{"rank": 1, "code": 605, "theme_ids": [...], "title": "..."},
         {"rank": 2, "code": 303, ...}, ...]
    """
    party_row = cmp_manifest[cmp_manifest["party"] == party]
    if party_row.empty:
        return []
    
    cmp_codes = []
    for rank in range(1, 6):  # Top 5
        code_col = f"code_{rank}"
        title_col = f"code_{rank}_title"
        theme_ids_col = f"code_{rank}_theme_ids"
        
        if code_col not in party_row.columns:
            continue
        
        code = party_row[code_col].values[0]
        title = party_row[title_col].values[0] if title_col in party_row.columns else "Unknown"
        theme_ids_str = party_row[theme_ids_col].values[0] if theme_ids_col in party_row.columns else ""
        
        # Parse theme_ids (semicolon-separated, e.g., "theme_0013;;theme_0018;;")
        theme_ids = [t.strip() for t in theme_ids_str.split(";;") if t.strip()]
        
        cmp_codes.append({
            "rank": rank,
            "code": code,
            "title": title,
            "theme_ids": theme_ids
        })
    
    return cmp_codes


def find_debates_for_themes(
    themes: List[str],
    debates_csv: str = "/scratch-shared/lsaleh/debates/debates.csv"
) -> List[str]:
    """
    Find debate document_ids that contain any of the given theme_ids.
    
    Returns sorted list of unique debate_ids.
    """
    df = pd.read_csv(debates_csv)
    
    # Filter rows where theme_id matches any in the themes list
    matching = df[df["theme_id"].isin(themes)]
    debate_ids = sorted(matching["dc_identifier"].unique().tolist())
    
    return debate_ids


def load_debate_interventions(
    debate_id: str,
    data_dir: str = "/scratch-shared/lsaleh/debates"
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load split intervention CSV for a debate.
    
    Tries multiple locations under /scratch-shared/lsaleh/debates/{year}/{debate_id}.csv.
    """
    # Try exact path first
    if os.path.exists(debate_id):
        csv_path = debate_id
    # Try a direct year folder first (debate ids are year-stamped)
    else:
        year_match = re.search(r"(19|20)\d{2}", debate_id)
        if year_match:
            year = year_match.group(0)
            year_path = f"{data_dir}/{year}/{debate_id}.csv"
            if os.path.exists(year_path):
                csv_path = year_path
            else:
                csv_path = None
        else:
            csv_path = None
    # Search in data_dir recursively
    if not csv_path:
        pattern = f"{data_dir}/**/{debate_id}.csv"
        files = glob.glob(pattern, recursive=True)
        if not files:
            return None, None
        csv_path = files[0]
    
    if not os.path.exists(csv_path):
        return None, None

    try:
        return pd.read_csv(csv_path), csv_path
    except:
        return None, csv_path


def aggregate_party_speech(
    df: pd.DataFrame,
    party: str,
    speech_col: str = "speech"
) -> Tuple[str, int]:
    """
    Aggregate all speeches by a party in a debate.
    
    Returns:
        (aggregated_text, total_word_count)
    """
    # Normalize party name
    party = normalize_party_name(party)
    
    party_df = df[df["party"] == party]
    
    if party_df.empty:
        return "", 0
    
    # Concatenate all speeches
    speeches = party_df[speech_col].fillna("").tolist()
    full_text = " ".join(speeches)
    
    # Count words (approximate)
    word_count = len(full_text.split())
    
    return full_text, word_count


# ============================================================================
# File & Output Management
# ============================================================================

def get_output_path(
    party: str,
    debate_id: str,
    output_type: str,
    output_dir: str,
    suffix: str = ""
) -> str:
    """
    Generate output path for (party, debate_id).
    
    output_type: "extracted", "summarized", "normalized"
    """
    base_name = f"{party}_{debate_id}{suffix}"
    
    if output_type == "extracted":
        return os.path.join(output_dir, f"{base_name}_extracted.csv")
    elif output_type == "summarized":
        # Summaries are debate-wide and should be saved as {debate_id}_summarized.csv
        return os.path.join(output_dir, f"{debate_id}_summarized.csv")
    elif output_type == "normalized":
        return os.path.join(output_dir, f"{base_name}_normalized.csv")
    else:
        return os.path.join(output_dir, f"{base_name}.csv")


def output_exists(
    party: str,
    debate_id: str,
    output_dir: str
) -> bool:
    """Check if normalized output already exists (indicates completion)."""
    path = get_output_path(party, debate_id, "normalized", output_dir)
    return os.path.exists(path)


def move_single_file(source_dir: str, pattern: str, target_path: str) -> bool:
    """Move the first file matching pattern from source_dir to target_path."""
    matches = sorted(glob.glob(os.path.join(source_dir, pattern)))
    if not matches:
        return False
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    os.replace(matches[0], target_path)
    return True


# ============================================================================
# Pipeline Execution
# ============================================================================

def run_command(cmd: List[str], description: str) -> bool:
    """Run shell command and log result."""
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with exit code {result.returncode}")
        return False
    
    print(f"\n[SUCCESS] {description} completed")
    return True


def run_extraction(
    text: str,
    output_path: str,
    party: str,
    debate_id: str,
    model_7b: str,
    logger: ProcessingLogger
) -> bool:
    """
    Run extraction on aggregated debate text.
    
    Creates a temporary CSV with single row and processes it.
    """
    output_dir = os.path.dirname(output_path)
    temp_dir = os.path.join(output_dir, f".tmp_extract_{debate_id}")
    os.makedirs(temp_dir, exist_ok=True)

    # Create temp input CSV
    temp_input = os.path.join(temp_dir, "input.csv")
    temp_df = pd.DataFrame({
        "document_id": ["temp"],
        "intervention_id": [1],
        "party": [party],
        "speech": [text],
        "n_words": [len(text.split())]
    })
    temp_df.to_csv(temp_input, index=False)
    
    cmd = [
        "python", "extract.py",
        "--input_csv", temp_input,
        "--output_dir", temp_dir,
        "--party", party,
        "--model_name", model_7b,
        "--target_party", party
    ]
    
    success = run_command(cmd, f"Extract for {party}")
    
    if not success:
        return False

    if not move_single_file(temp_dir, f"{party}_claims*.csv", output_path):
        logger.log(f"Extraction output not found in {temp_dir}", level="ERROR")
        return False

    # Remove temp dir contents if possible
    try:
        for f in glob.glob(os.path.join(temp_dir, "*.csv")):
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(temp_input):
            os.remove(temp_input)
        if os.path.isdir(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except:
        pass
    
    return success


def run_summarization(
    debate_input_csv: str,
    debate_id: str,
    summary_path: str,
    model_7b: str,
    logger: ProcessingLogger
) -> bool:
    """
    Run unbiased summarization on full debate (debate_input_csv) and save
    resulting summary as `summary_path` (e.g., {debate_id}_summarized.csv).

    This calls `incr_summary.py` with the debate CSV as input and then
    moves/renames the produced file into the canonical `summary_path` so
    it is independent of CMP codes or parties.
    """
    out_dir = os.path.dirname(summary_path)
    temp_dir = os.path.join(out_dir, f".tmp_summary_{debate_id}")
    os.makedirs(temp_dir, exist_ok=True)

    cmd = [
        "python", "incr_summary.py",
        "--input_csv", debate_input_csv,
        "--output_dir", temp_dir,
        "--model_name", model_7b
    ]

    if not run_command(cmd, f"Summarize debate {debate_id}"):
        return False

    if not move_single_file(temp_dir, f"{debate_id}_summary*.csv", summary_path):
        logger.log(f"No summary file produced in {temp_dir}", level="ERROR")
        return False
    try:
        if os.path.isdir(temp_dir):
            for f in glob.glob(os.path.join(temp_dir, "*.csv")):
                if os.path.exists(f):
                    os.remove(f)
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        return True
    except Exception as e:
        logger.log(f"Failed to clean summary temp dir {temp_dir}: {e}", level="ERROR")
        return False


def run_normalization(
    extracted_csv: str,
    summarized_csv: str,
    output_path: str,
    debates_csv: str,
    party: str,
    model_32b: str,
    logger: ProcessingLogger
) -> bool:
    """Run normalization on extracted and summarized claims."""
    output_dir = os.path.dirname(output_path)
    temp_dir = os.path.join(output_dir, f".tmp_normalize_{os.path.basename(output_path).replace('.csv', '')}")
    os.makedirs(temp_dir, exist_ok=True)

    cmd = [
        "python", "normalize.py",
        "--claims_csv", extracted_csv,
        "--debates_csv", debates_csv,
        "--summaries_csv", summarized_csv,
        "--output_dir", temp_dir,
        "--party", party,
        "--model_name", model_32b
    ]
    
    if not run_command(cmd, f"Normalize for {party}"):
        return False

    if not move_single_file(temp_dir, f"{party}_normalized.csv", output_path):
        logger.log(f"Normalization output not found in {temp_dir}", level="ERROR")
        return False

    try:
        for f in glob.glob(os.path.join(temp_dir, "*.csv")):
            if os.path.exists(f):
                os.remove(f)
        if os.path.isdir(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except:
        pass

    return True


# ============================================================================
# Main Pipeline
# ============================================================================

def process_party(
    party: str,
    model_7b: str,
    model_32b: str,
    min_tokens: int = 100,
    extract_dir: str = "/scratch-shared/lsaleh/debates/extracted",
    summary_dir: str = "/scratch-shared/lsaleh/debates/summaries",
    normalize_dir: str = "/scratch-shared/lsaleh/debates/normalized",
    debates_csv: str = "/scratch-shared/lsaleh/debates/debates.csv",
    cmp_manifest_csv: str = "/scratch-shared/lsaleh/debates/cmp_manifest.csv",
    data_dir: str = "/scratch-shared/lsaleh/debates",
    resume: bool = False,
    metadata_csv: str = "outputs/debate_cmp_metadata.csv"
):
    """
    Process a single party through the full pipeline.
    
    For each of the party's top 5 CMP codes:
    - Find relevant debates
    - Filter by token threshold
    - Run extract → summarize → normalize
    - Save outputs as (party, cmp_code, debate_id)
    """
    
    # Setup logging
    logger = ProcessingLogger(party)
    logger.log(f"Starting pipeline for party: {party}")
    logger.log(f"  Model 7B: {model_7b}")
    logger.log(f"  Model 32B: {model_32b}")
    logger.log(f"  Min tokens threshold: {min_tokens}")
    logger.log(f"  Data directory: {data_dir}")
    
    # Create output directories
    for d in [extract_dir, summary_dir, normalize_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Load checkpoint if resuming
    completed = set()
    if resume:
        completed = logger.load_checkpoint()
        logger.log(f"Resuming: {len(completed)} debate_ids already completed")
    
    # Load CMP manifest
    try:
        cmp_manifest = load_cmp_manifest(cmp_manifest_csv)
    except Exception as e:
        logger.log(f"ERROR loading CMP manifest: {e}", level="ERROR")
        return
    
    # Get party's top 5 CMP codes
    cmp_codes = get_party_cmp_codes(cmp_manifest, party)
    if not cmp_codes:
        logger.log(f"Party '{party}' not found in CMP manifest", level="ERROR")
        return
    
    logger.log(f"Found {len(cmp_codes)} CMP codes for {party}")
    
    # Track statistics
    stats = {
        "processed": 0,
        "skipped_no_file": 0,
        "skipped_no_interventions": 0,
        "skipped_below_threshold": 0,
        "skipped_already_done": 0,
        "completed": 0,
        "failed": 0
    }
    
    # Main loop: iterate over CMP codes
    for cmp_info in cmp_codes:
        cmp_rank = cmp_info["rank"]
        cmp_code = cmp_info["code"]
        theme_ids = cmp_info["theme_ids"]
        
        logger.log(f"\n--- Processing CMP Rank {cmp_rank}: Code {cmp_code} ({cmp_info['title']}) ---")
        logger.log(f"    Theme IDs: {', '.join(theme_ids)}")
        
        # Find debates with these themes
        try:
            debate_ids = find_debates_for_themes(theme_ids, debates_csv)
            logger.log(f"    Found {len(debate_ids)} debates matching these themes")
        except Exception as e:
            logger.log(f"    ERROR finding debates: {e}", level="ERROR")
            continue
        
        if not debate_ids:
            logger.log(f"    No debates found for CMP rank {cmp_rank}")
            continue

        # Persist mapping metadata for this party / cmp combo up front
        metadata_rows = build_debate_metadata_rows(party, cmp_info, debate_ids, debates_csv)
        write_metadata_csv(metadata_rows, metadata_csv)
        logger.log(f"    Metadata updated: {len(metadata_rows)} debate rows -> {metadata_csv}")
        
        # Process each debate
        for debate_id in tqdm(debate_ids, desc=f"CMP Rank {cmp_rank}"):
            # Check if already done
            if debate_id in completed:
                logger.log_skipped(cmp_rank, cmp_code, debate_id, "already_completed")
                stats["skipped_already_done"] += 1
                continue
            
            # Load debate interventions (also get CSV path)
            interventions, debate_csv_path = load_debate_interventions(debate_id, data_dir)
            if interventions is None:
                logger.log_skipped(cmp_rank, cmp_code, debate_id, "file_not_found")
                stats["skipped_no_file"] += 1
                continue

            # Aggregate party's speech
            debate_text, word_count = aggregate_party_speech(interventions, party)
            
            if not debate_text or word_count == 0:
                logger.log_skipped(cmp_rank, cmp_code, debate_id, "party_has_no_interventions")
                stats["skipped_no_interventions"] += 1
                continue
            
            # Check token threshold
            if word_count < min_tokens:
                logger.log_skipped(
                    cmp_rank, cmp_code, debate_id, 
                    f"below_threshold(words={word_count}, min={min_tokens})"
                )
                stats["skipped_below_threshold"] += 1
                continue
            
            logger.log(f"  Processing: debate_id={debate_id}, words={word_count}")
            stats["processed"] += 1
            
            # Generate output paths
            extracted_path = get_output_path(party, debate_id, "extracted", extract_dir)
            summarized_path = get_output_path(party, debate_id, "summarized", summary_dir)
            normalized_path = get_output_path(party, debate_id, "normalized", normalize_dir)

            # Step 1: Summarize debate (unbiased) — one per debate, saved as {debate_id}_summarized.csv
            logger.log(f"    [1/3] Ensuring debate summary exists...")
            if not os.path.exists(summarized_path):
                # If we have the original debate CSV path, use it; otherwise create a temp CSV
                if debate_csv_path:
                    debate_input_csv = debate_csv_path
                    temp_created = False
                else:
                    # Create a temporary debate CSV with aggregated text
                    debate_input_csv = ".temp_debate_for_summary.csv"
                    temp_df = pd.DataFrame({
                        "document_id": [debate_id],
                        "intervention_id": [1],
                        "party": [party],
                        "speech": [debate_text],
                        "n_words": [len(debate_text.split())]
                    })
                    temp_df.to_csv(debate_input_csv, index=False)
                    temp_created = True

                if not run_summarization(debate_input_csv, debate_id, summarized_path, model_7b, logger):
                    logger.log(f"    FAILED: summarization", level="ERROR")
                    stats["failed"] += 1
                    # Clean up temp if created
                    if 'temp_created' in locals() and temp_created and os.path.exists(debate_input_csv):
                        os.remove(debate_input_csv)
                    continue

                if 'temp_created' in locals() and temp_created and os.path.exists(debate_input_csv):
                    os.remove(debate_input_csv)
            else:
                logger.log(f"    [1/3] Using existing summarized file")

            # Step 2: Extract
            logger.log(f"    [2/3] Extracting claims...")
            if not os.path.exists(extracted_path):
                if not run_extraction(debate_text, extracted_path, party, debate_id, model_7b, logger):
                    logger.log(f"    FAILED: extraction", level="ERROR")
                    stats["failed"] += 1
                    continue
            else:
                logger.log(f"    [2/3] Using existing extracted file")
            
            # Check if extraction produced results
            try:
                extracted_df = pd.read_csv(extracted_path)
                if extracted_df.empty or len(extracted_df) == 0:
                    logger.log_skipped(cmp_rank, cmp_code, debate_id, "extraction_produced_no_claims")
                    stats["skipped_below_threshold"] += 1
                    continue
            except:
                logger.log(f"    FAILED: could not read extracted CSV", level="ERROR")
                stats["failed"] += 1
                continue
            
            # Step 3: Normalize
            logger.log(f"    [3/3] Normalizing claims...")
            if not run_normalization(
                extracted_path, summarized_path, normalized_path,
                debates_csv, party, model_32b, logger
            ):
                logger.log(f"    FAILED: normalization", level="ERROR")
                stats["failed"] += 1
                continue
            
            logger.log(f"    SUCCESS: completed (party={party}, cmp_code={cmp_code}, debate_id={debate_id})")
            completed.add(debate_id)
            stats["completed"] += 1
            
            # Save checkpoint after each successful completion
            logger.save_checkpoint(completed)
    
    # Final summary
    logger.log(f"\n{'='*80}")
    logger.log(f"FINAL SUMMARY FOR {party}")
    logger.log(f"{'='*80}")
    logger.log(f"  Processed:                {stats['processed']}")
    logger.log(f"  Completed successfully:   {stats['completed']}")
    logger.log(f"  Failed:                   {stats['failed']}")
    logger.log(f"  Skipped (already done):   {stats['skipped_already_done']}")
    logger.log(f"  Skipped (file not found): {stats['skipped_no_file']}")
    logger.log(f"  Skipped (no interventions): {stats['skipped_no_interventions']}")
    logger.log(f"  Skipped (below threshold): {stats['skipped_below_threshold']}")
    logger.log(f"{'='*80}")


# ============================================================================
# Argument Parsing & Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Party-centric batch pipeline: extract → summarize (unbiased) → normalize per CMP code"
    )
    
    # Required
    parser.add_argument("--party", type=str, required=True,
                        help="Party name (e.g., VVD, PvdA, PVV)")
    parser.add_argument("--model_7b", type=str, required=True,
                        help="7B model name for extract and summarize")
    parser.add_argument("--model_32b", type=str, required=True,
                        help="32B model name for normalize")
    
    # Thresholds
    parser.add_argument("--min_tokens", type=int, default=100,
                        help="Minimum word count threshold for processing (default: 100)")
    
    # Directories
    parser.add_argument("--extract_dir", type=str, default="/scratch-shared/lsaleh/debates/extracted",
                        help="Output directory for extraction")
    parser.add_argument("--summary_dir", type=str, default="/scratch-shared/lsaleh/debates/summaries",
                        help="Output directory for summaries (default: scratch shared summaries)")
    parser.add_argument("--normalize_dir", type=str, default="/scratch-shared/lsaleh/debates/normalized",
                        help="Output directory for normalized claims")
    parser.add_argument("--debates_csv", type=str, default="/scratch-shared/lsaleh/debates/debates.csv",
                        help="Path to debates.csv")
    parser.add_argument("--cmp_manifest_csv", type=str, default="/scratch-shared/lsaleh/debates/cmp_manifest.csv",
                        help="Path to cmp_manifest.csv")
    parser.add_argument("--data_dir", type=str, default="/scratch-shared/lsaleh/debates",
                        help="Root directory containing debate CSV files")
    
    # Control flags
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--metadata_csv", type=str, default="outputs/debate_cmp_metadata.csv",
                        help="Metadata CSV for party x cmp_rank x debate mappings")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    process_party(
        party=args.party,
        model_7b=args.model_7b,
        model_32b=args.model_32b,
        min_tokens=args.min_tokens,
        extract_dir=args.extract_dir,
        summary_dir=args.summary_dir,
        normalize_dir=args.normalize_dir,
        debates_csv=args.debates_csv,
        cmp_manifest_csv=args.cmp_manifest_csv,
        data_dir=args.data_dir,
        resume=args.resume,
        metadata_csv=args.metadata_csv
    )


if __name__ == "__main__":
    main()
