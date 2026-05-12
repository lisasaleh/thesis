"""
Party-Centric Batch Pipeline (STAGED): extract → summarize (unbiased) → normalize per CMP code

IMPROVED ARCHITECTURE:
Instead of loading models per debate, we batch process ALL debates for each CMP code together:

For each CMP code rank (1-5):
  1. Collect ALL debates matching that CMP's theme_ids into ONE input CSV
  2. Call extract.py ONCE on full CSV (model loaded once, all debates processed)
  3. Call incr_summary.py ONCE on full CSV (model loaded once)
  4. Call normalize.py ONCE with combined extract+summary outputs
  5. Save metadata mapping party/cmp_code/debate_id for tracking

This eliminates the per-debate model reloading that made the original pipeline inefficient.

Usage:
    python batch_all_party_centric_staged.py --party VVD \\
        --model_7b Qwen/Qwen2.5-7B-Instruct \\
        --model_32b /path/to/Qwen2.5-32B-Instruct
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


# Party name aliases
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
    
    def save_checkpoint(self, completed_cmp_ranks: Set[int]):
        """Save completed CMP ranks."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                "completed_cmp_ranks": sorted(list(completed_cmp_ranks))
            }, f, indent=2)
    
    def load_checkpoint(self) -> Set[int]:
        """Load completed CMP ranks."""
        if not os.path.exists(self.checkpoint_file):
            return set()
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                return set(item for item in data.get("completed_cmp_ranks", []))
        except:
            return set()


# ============================================================================
# Data Loading & Theme Matching
# ============================================================================

def load_cmp_manifest(csv_path: str) -> pd.DataFrame:
    """Load CMP manifest with party → top 5 CMP codes mapping."""
    return pd.read_csv(csv_path)


def normalize_party_name(party: str) -> str:
    """Normalize party name using aliases."""
    return PARTY_ALIASES.get(party, party)


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
    debates_csv: str
) -> List[str]:
    """
    Find debate document_ids that contain any of the given theme_ids.
    Excludes debates with day_count == -1 (out of date range).
    
    Returns sorted list of unique debate_ids.
    """
    df = pd.read_csv(debates_csv)
    
    # Filter rows where theme_id matches any in the themes list
    matching = df[df["theme_id"].isin(themes)]
    
    # Exclude debates with day_count == -1 (out of date range)
    matching = matching[matching["day_count"] != -1]
    
    debate_ids = sorted(matching["dc_identifier"].unique().tolist())
    
    return debate_ids


def load_debate_interventions(
    debate_id: str,
    data_dir: str
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load split intervention CSV for a debate.
    
    Returns (dataframe, file_path) or (None, None) if not found.
    """
    # Try exact path first
    if os.path.exists(debate_id):
        csv_path = debate_id
    # Try year-based folder
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
    
    # Search recursively if not found
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
    """Aggregate all speeches by a party in a debate."""
    party = normalize_party_name(party)
    party_df = df[df["party"] == party]
    
    if party_df.empty:
        return "", 0
    
    speeches = party_df[speech_col].fillna("").tolist()
    full_text = " ".join(speeches)
    word_count = len(full_text.split())
    
    return full_text, word_count


# ============================================================================
# Batch Input Creation
# ============================================================================

def build_cmp_batch_input_csv(
    party: str,
    cmp_info: Dict,
    debate_ids: List[str],
    debates_csv: str,
    data_dir: str,
    output_csv: str,
    min_tokens: int = 100,
    logger: Optional[ProcessingLogger] = None
) -> Tuple[int, int]:
    """
    Build a batch input CSV for all debates in this CMP code.
    
    For each debate_id:
      - Load debate interventions CSV
      - Aggregate party's speech
      - Only include if >= min_tokens
    
    Returns:
      (num_included_debates, num_skipped_debates)
    """
    rows = []
    skipped = 0
    
    # Load debates metadata for date info
    debates_df = pd.read_csv(debates_csv)
    debate_lookup = debates_df.drop_duplicates(subset=["dc_identifier"]).set_index("dc_identifier")
    
    for debate_id in tqdm(debate_ids, desc=f"Building input for CMP Rank {cmp_info['rank']} ({cmp_info['code']})"):
        # Load debate interventions
        interventions, _ = load_debate_interventions(debate_id, data_dir)
        if interventions is None:
            skipped += 1
            if logger:
                logger.log(f"  Skipped {debate_id}: file not found", level="DEBUG")
            continue
        
        # Aggregate party speech
        party_text, word_count = aggregate_party_speech(interventions, party)
        
        if not party_text or word_count < min_tokens:
            skipped += 1
            if logger:
                logger.log(f"  Skipped {debate_id}: {word_count} words < {min_tokens}", level="DEBUG")
            continue
        
        # Get debate date
        date_value = ""
        if debate_id in debate_lookup.index:
            for col in ("foi_meetingDate", "dc_date", "date"):
                if col in debate_lookup.columns and pd.notna(debate_lookup.loc[debate_id, col]):
                    date_value = str(debate_lookup.loc[debate_id, col]).strip()
                    break
        
        # Create row (same schema as original debate interventions)
        row = {
            "document_id": debate_id,
            "intervention_id": 1,  # Aggregated as single intervention
            "party": party,
            "speaker": party,  # For batch, speaker is the party aggregate
            "speaker_label": party,  # For batch, no individual speaker label
            "speech": party_text,
            "n_words": word_count,
            "date": date_value,
        }
        rows.append(row)
    
    # Write batch input CSV
    if rows:
        batch_df = pd.DataFrame(rows)
        batch_df.to_csv(output_csv, index=False)
        if logger:
            logger.log(f"  Created batch input CSV: {output_csv} ({len(rows)} debates)")
    else:
        if logger:
            logger.log(f"  No valid debates for batch input")
    
    return len(rows), skipped


# ============================================================================
# File & Output Management
# ============================================================================

def run_command(cmd: List[str], description: str, fatal: bool = False) -> bool:
    """
    Run shell command and log result.
    
    Args:
        cmd: Command to run
        description: Description of command
        fatal: If False, continue on error; if True, treat as fatal failure
    """
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        msg = f"[WARNING] {description} failed with exit code {result.returncode}"
        if fatal:
            print(f"\n[FATAL] {msg}")
        else:
            print(f"\n{msg} - CONTINUING ANYWAY")
        return False
    
    print(f"\n[SUCCESS] {description} completed")
    return True


def find_latest_file(directory: str, pattern: str) -> Optional[str]:
    """Find most recent file matching pattern in directory."""
    if not os.path.exists(directory):
        return None
    
    files = [f for f in os.listdir(directory) if pattern in f]
    if not files:
        return None
    
    files.sort(reverse=True)
    return os.path.join(directory, files[0])


def concatenate_summary_csvs(directory: str, output_csv: str) -> bool:
    """
    Concatenate all *_summary*.csv files in a directory into one combined CSV.
    
    incr_summary.py writes one file per document, but normalization needs
    all summaries in one file for lookup.
    
    Args:
        directory: Directory containing summary CSV files
        output_csv: Path to write combined CSV
    
    Returns:
        True if successful, False if no files found
    """
    if not os.path.exists(directory):
        return False
    
    csv_files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith("_summary.csv") or (f.endswith(".csv") and "summary" in f)
    ])
    
    if not csv_files:
        return False
    
    try:
        dfs = [pd.read_csv(f) for f in csv_files]
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(output_csv, index=False)
        print(f"[DEBUG] Concatenated {len(csv_files)} summary files into {output_csv}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to concatenate summary CSVs: {e}")
        return False


# ============================================================================
# Main Pipeline
# ============================================================================

def process_party_staged(
    party: str,
    model_7b: str,
    model_32b: str,
    min_tokens: int = 100,
    extract_dir: str = "/scratch-shared/lsaleh/extracted",
    summary_dir: str = "/scratch-shared/lsaleh/summaries",
    normalize_dir: str = "/scratch-shared/lsaleh/normalized",
    debates_csv: str = "outputs/debates.csv",
    cmp_manifest_csv: str = "outputs/cmp_manifest.csv",
    data_dir: str = "/scratch-shared/lsaleh/debates/",
    resume: bool = False,
    metadata_csv: str = "outputs/debate_cmp_metadata.csv"
):
    """
    Process a single party through the STAGED pipeline.
    
    For each CMP code rank:
      1. Create batch input CSV with all debates for that CMP
      2. Call extract.py ONCE on batch (loads model once)
      3. Call incr_summary.py ONCE on batch
      4. Call normalize.py ONCE on combined extract+summary
    """
    
    # Setup logging
    logger = ProcessingLogger(party)
    logger.log(f"Starting STAGED pipeline for party: {party}")
    logger.log(f"  Model 7B: {model_7b}")
    logger.log(f"  Model 32B: {model_32b}")
    logger.log(f"  Min tokens: {min_tokens}")
    logger.log(f"  Data directory: {data_dir}")
    
    # Create output directories
    for d in [extract_dir, summary_dir, normalize_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Load checkpoint if resuming
    completed_ranks = set()
    if resume:
        completed_ranks = logger.load_checkpoint()
        logger.log(f"Resuming: CMP ranks {sorted(completed_ranks)} already completed")
    
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
        "cmp_ranks_processed": 0,
        "total_debates_batch_input": 0,
        "total_debates_skipped": 0,
        "cmp_ranks_completed": 0,
        "cmp_ranks_failed": 0
    }
    
    # Main loop: iterate over CMP codes
    for cmp_info in cmp_codes:
        cmp_rank = cmp_info["rank"]
        cmp_code = cmp_info["code"]
        theme_ids = cmp_info["theme_ids"]
        
        # Skip if already completed
        if cmp_rank in completed_ranks:
            logger.log(f"Skipping CMP Rank {cmp_rank}: already completed")
            continue
        
        logger.log(f"\n{'='*80}")
        logger.log(f"Processing CMP Rank {cmp_rank}: Code {cmp_code} ({cmp_info['title']})")
        logger.log(f"Theme IDs: {', '.join(theme_ids)}")
        logger.log(f"{'='*80}")
        
        # Find debates with these themes
        try:
            debate_ids = find_debates_for_themes(theme_ids, debates_csv)
            logger.log(f"Found {len(debate_ids)} debates matching these themes (day_count != -1, i.e., within date range)")
        except Exception as e:
            logger.log(f"ERROR finding debates: {e}", level="ERROR")
            stats["cmp_ranks_failed"] += 1
            continue
        
        if not debate_ids:
            logger.log(f"No debates found for CMP rank {cmp_rank}")
            stats["cmp_ranks_failed"] += 1
            continue
        
        stats["cmp_ranks_processed"] += 1
        
        # Step 1: Build batch input CSV
        temp_batch_input = os.path.join(".tmp_batch", f"{party}_cmp_rank_{cmp_rank}_input.csv")
        os.makedirs(os.path.dirname(temp_batch_input), exist_ok=True)
        
        num_included, num_skipped = build_cmp_batch_input_csv(
            party, cmp_info, debate_ids, debates_csv, data_dir,
            temp_batch_input, min_tokens, logger
        )

        stats["total_debates_batch_input"] += num_included
        stats["total_debates_skipped"] += num_skipped
        
        if num_included == 0:
            logger.log(f"No valid debates for CMP rank {cmp_rank} after filtering")
            stats["cmp_ranks_failed"] += 1
            if os.path.exists(temp_batch_input):
                os.remove(temp_batch_input)
            continue
        
        # Step 2: Extract (batch) - load model ONCE for all debates
        logger.log(f"\n[1/3] Extraction (batch of {num_included} debates)...")
        temp_extract_dir = os.path.join(".tmp_batch", f"extract_cmp_{cmp_rank}")
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        # NOTE: 7B model loads here, then reloads again in summarization (Step 3)
        # Future optimization: Refactor to load 7B once and run extract+summary sequentially
        # in the same Python process to avoid model reload overhead.
        extract_cmd = [
            "python", "extract.py",
            "--input_csv", temp_batch_input,
            "--output_dir", temp_extract_dir,
            "--party", party,
            "--model_name", model_7b,
            "--target_party", party
        ]
        
        extract_success = run_command(extract_cmd, f"Extract CMP Rank {cmp_rank}", fatal=False)
        
        # Find extracted output - if extraction failed, try to continue anyway
        extract_output = find_latest_file(temp_extract_dir, f"{party}_claims") if extract_success else None
        
        if not extract_output:
            logger.log(f"WARNING: no extract output found in {temp_extract_dir}", level="WARN")
            logger.log(f"  → This CMP rank will have no extracted claims")
            extract_output = None  # Will skip normalization
        else:
            logger.log(f"  ✓ Found extraction output: {extract_output}")
        
        # Step 3: Summarize (batch) - load model ONCE for all debates
        logger.log(f"\n[2/3] Summarization (batch of {num_included} debates)...")
        temp_summary_dir = os.path.join(".tmp_batch", f"summary_cmp_{cmp_rank}")
        os.makedirs(temp_summary_dir, exist_ok=True)
        
        # NOTE: 7B model loads here (already loaded once in extract, Step 2)
        # Model reload is necessary because extract.py runs in a separate Python process
        # To avoid this reload, combine extract+summary into a single pipeline in the same process
        summary_cmd = [
            "python", "incr_summary.py",
            "--input_csv", temp_batch_input,
            "--output_dir", temp_summary_dir,
            "--model_name", model_7b
        ]
        
        summary_success = run_command(summary_cmd, f"Summarize CMP Rank {cmp_rank}", fatal=False)
        
        # Concatenate all per-document summary files into one combined summary CSV
        combined_summary_output = os.path.join(temp_summary_dir, "combined_summary.csv")
        summary_concat_success = concatenate_summary_csvs(temp_summary_dir, combined_summary_output)
        
        if not summary_concat_success:
            logger.log(f"WARNING: Failed to concatenate summary files in {temp_summary_dir}", level="WARN")
            summary_output = None
        else:
            logger.log(f"  ✓ Combined {len([f for f in os.listdir(temp_summary_dir) if 'summary.csv' in f])} summary files: {combined_summary_output}")
            summary_output = combined_summary_output
        
        # Step 4: Normalize (batch) - only run if we have both extract and summary outputs
        logger.log(f"\n[3/3] Normalization (batch of {num_included} debates)...")
        
        if not extract_output or not summary_output:
            logger.log(f"SKIPPING normalization: missing extract ({extract_output is not None}) or summary ({summary_output is not None})", level="WARN")
            stats["cmp_ranks_failed"] += 1
            # Cleanup
            for temp_d in [temp_extract_dir, temp_summary_dir]:
                try:
                    import shutil
                    if os.path.exists(temp_d):
                        shutil.rmtree(temp_d)
                except:
                    pass
            continue
        
        temp_normalize_dir = os.path.join(".tmp_batch", f"normalize_cmp_{cmp_rank}")
        os.makedirs(temp_normalize_dir, exist_ok=True)
        
        normalize_cmd = [
            "python", "normalize.py",
            "--claims_csv", extract_output,
            "--debates_csv", temp_batch_input,
            "--summaries_csv", summary_output,
            "--output_dir", temp_normalize_dir,
            "--party", party,
            "--model_name", model_32b
        ]
        
        normalize_success = run_command(normalize_cmd, f"Normalize CMP Rank {cmp_rank}", fatal=False)
        
        # Find normalized output
        normalize_output = find_latest_file(temp_normalize_dir, f"{party}_normalized") if normalize_success else None
        
        if not normalize_output:
            logger.log(f"WARNING: no normalize output found in {temp_normalize_dir}", level="WARN")
            stats["cmp_ranks_failed"] += 1
            for temp_d in [temp_extract_dir, temp_summary_dir, temp_normalize_dir]:
                try:
                    import shutil
                    if os.path.exists(temp_d):
                        shutil.rmtree(temp_d)
                except:
                    pass
            continue
        
        # Step 5: Move outputs to final directories
        logger.log(f"\nMoving outputs to final directories...")
        
        # Move extracted
        if extract_output:
            extracted_final = os.path.join(extract_dir, f"{party}_cmp_{cmp_rank}_claims.csv")
            os.makedirs(os.path.dirname(extracted_final), exist_ok=True)
            os.replace(extract_output, extracted_final)
            logger.log(f"  → {extracted_final}")
        
        # Move summary
        if summary_output:
            summary_final = os.path.join(summary_dir, f"{party}_cmp_{cmp_rank}_summary.csv")
            os.makedirs(os.path.dirname(summary_final), exist_ok=True)
            os.replace(summary_output, summary_final)
            logger.log(f"  → {summary_final}")
        
        # Move normalized
        normalized_final = os.path.join(normalize_dir, f"{party}_cmp_{cmp_rank}_normalized.csv")
        os.makedirs(os.path.dirname(normalized_final), exist_ok=True)
        os.replace(normalize_output, normalized_final)
        logger.log(f"  → {normalized_final}")
        
        logger.log(f"SUCCESS: CMP Rank {cmp_rank} pipeline completed")
        completed_ranks.add(cmp_rank)
        stats["cmp_ranks_completed"] += 1
        logger.save_checkpoint(completed_ranks)
        
        # Cleanup temp directories for this CMP
        for temp_d in [temp_extract_dir, temp_summary_dir, temp_normalize_dir]:
            try:
                import shutil
                if os.path.exists(temp_d):
                    shutil.rmtree(temp_d)
            except:
                pass
    
    # Final summary
    logger.log(f"\n{'='*80}")
    logger.log(f"FINAL SUMMARY FOR {party}")
    logger.log(f"{'='*80}")
    logger.log(f"  CMP Ranks Processed:        {stats['cmp_ranks_processed']}")
    logger.log(f"  CMP Ranks Completed:        {stats['cmp_ranks_completed']} ✓")
    logger.log(f"  CMP Ranks Failed:           {stats['cmp_ranks_failed']} ✗")
    logger.log(f"  Total Debates in Batches:   {stats['total_debates_batch_input']}")
    logger.log(f"  Total Debates Skipped:      {stats['total_debates_skipped']}")
    logger.log(f"{'='*80}")


# ============================================================================
# Argument Parsing & Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Party-centric STAGED batch pipeline: all debates per CMP processed together"
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
                        help="Minimum word count threshold (default: 100)")
    
    # Directories
    parser.add_argument("--extract_dir", type=str, default="/scratch-shared/lsaleh/extracted",
                        help="Output directory for extraction")
    parser.add_argument("--summary_dir", type=str, default="/scratch-shared/lsaleh/summaries",
                        help="Output directory for summaries")
    parser.add_argument("--normalize_dir", type=str, default="/scratch-shared/lsaleh/normalized",
                        help="Output directory for normalized claims")
    parser.add_argument("--debates_csv", type=str, default="outputs/debates.csv",
                        help="Path to debates.csv")
    parser.add_argument("--cmp_manifest_csv", type=str, default="outputs/cmp_manifest.csv",
                        help="Path to cmp_manifest.csv")
    parser.add_argument("--data_dir", type=str, default="/scratch-shared/lsaleh/debates/",
                        help="Root directory containing debate CSV files")
    
    # Control flags
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    process_party_staged(
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
    )


if __name__ == "__main__":
    main()
