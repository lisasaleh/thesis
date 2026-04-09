"""
Batch orchestrator for the full pipeline: extract → summarize → normalize → cluster → select

Usage:
    python batch_all.py --input_csv data/VVD_debat.csv --party VVD --model_7b Qwen/Qwen2.5-7B-Instruct --model_32b /path/to/Qwen2.5-32B-Instruct --selection_model_name klue/bert-base
"""

import os
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and handle errors."""
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch orchestrator for extract → summarize → normalize → cluster → select pipeline"
    )
    
    # Required arguments
    parser.add_argument("--input_csv", type=str, required=True, 
                        help="Input debate CSV (e.g., data/VVD_debat.csv)")
    parser.add_argument("--party", type=str, required=True,
                        help="Party name (used for filtering and output naming)")
    
    # Model arguments
    parser.add_argument("--model_7b", type=str, required=True,
                        help="7B model name for extract and summarize")
    parser.add_argument("--model_32b", type=str, required=True,
                        help="32B model name for normalize")
    
    # Output directories
    parser.add_argument("--extract_dir", type=str, default="outputs/extracted",
                        help="Output directory for extraction")
    parser.add_argument("--summary_dir", type=str, default="outputs/summaries",
                        help="Output directory for summaries")
    parser.add_argument("--normalize_dir", type=str, default="outputs/normalized",
                        help="Output directory for normalized claims")
    parser.add_argument("--cluster_dir", type=str, default="outputs/clustering",
                        help="Output directory for clustering results")
    parser.add_argument("--selection_dir", type=str, default="outputs/selection",
                        help="Output directory for selection results")
    
    # Optional flags
    parser.add_argument("--add_timestamp", action="store_true",
                        help="Add timestamp to output filenames")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed step")
    parser.add_argument("--skip_extract", action="store_true", help="Skip extraction step")
    parser.add_argument("--skip_summarize", action="store_true", help="Skip summarization step")
    parser.add_argument("--skip_normalize", action="store_true", help="Skip normalization step")
    parser.add_argument("--skip_cluster", action="store_true", help="Skip clustering step")
    parser.add_argument("--skip_selection", action="store_true", help="Skip selection step")
    
    # Clustering parameters
    parser.add_argument("--min_cluster_size", type=int, default=3)
    parser.add_argument("--min_samples", type=int, default=1)
    parser.add_argument("--n_neighbors", type=int, default=8)
    parser.add_argument("--min_dist", type=float, default=0.0)
    
    # Selection parameters
    parser.add_argument("--selection_model_name", type=str, default="microsoft/deberta-base",
                        help="HF model for pairwise match classification")
    parser.add_argument("--selection_threshold", type=float, default=0.5,
                        help="Probability threshold for positive match in selection")
    parser.add_argument("--selection_batch_size", type=int, default=16,
                        help="Batch size for selection model inference")
    parser.add_argument("--keep_cluster_metadata", action="store_true",
                        help="Keep original cluster metadata in selection output")
    
    return parser.parse_args()


def get_output_filename(base_csv, output_dir, suffix, timestamp_str=""):
    """Generate output filename."""
    base_name = Path(base_csv).stem  # e.g., VVD_debat
    filename = f"{base_name}_{suffix}{timestamp_str}.csv"
    return os.path.join(output_dir, filename)


def find_latest_file(directory, pattern):
    """Find the most recent file matching a pattern."""
    if not os.path.exists(directory):
        return None
    
    files = [f for f in os.listdir(directory) if pattern in f]
    if not files:
        return None
    
    files.sort(reverse=True)  # Most recent first
    return os.path.join(directory, files[0])


def main():
    args = parse_args()
    
    # Setup
    timestamp_str = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if args.add_timestamp else ""
    checkpoint_file = f".batch_checkpoint_{args.party}.json"
    
    # Load checkpoint if resuming
    completed_steps = set()
    if args.resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            completed_steps = set(checkpoint.get("completed_steps", []))
        print(f"[INFO] Resuming from checkpoint. Completed steps: {completed_steps}")
    
    # Define pipeline steps
    steps = []
    
    # Step 1: Extract
    if not args.skip_extract and "extract" not in completed_steps:
        extract_output = get_output_filename(args.input_csv, args.extract_dir, "claims", timestamp_str)
        steps.append({
            "name": "extract",
            "cmd": [
                "python", "extract.py",
                "--input_csv", args.input_csv,
                "--output_dir", args.extract_dir,
                "--output_claims_csv", extract_output,
                "--model_name", args.model_7b,
                "--target_party", args.party,
                *(["--add_timestamp"] if args.add_timestamp else []),
                *(["--resume"] if args.resume else []),
            ],
            "output": extract_output
        })
    
    # Step 2: Incremental Summary
    if not args.skip_summarize and "summarize" not in completed_steps:
        summary_output = get_output_filename(args.input_csv, args.summary_dir, "summary", timestamp_str)
        steps.append({
            "name": "summarize",
            "cmd": [
                "python", "incr_summary.py",
                "--input_csv", args.input_csv,
                "--output_dir", args.summary_dir,
                "--model_name", args.model_7b,
                *(["--add_timestamp"] if args.add_timestamp else []),
                *(["--resume"] if args.resume else []),
            ],
            "output": summary_output
        })
    
    # Step 3: Normalize
    if not args.skip_normalize and "normalize" not in completed_steps:
        # Find latest extract and summary outputs
        extract_output = find_latest_file(args.extract_dir, f"{Path(args.input_csv).stem}_claims")
        summary_output = find_latest_file(args.summary_dir, f"{Path(args.input_csv).stem}_summary")
        
        if not extract_output or not summary_output:
            print("[ERROR] Cannot find extract or summary outputs for normalize step")
            return
        
        normalize_output = get_output_filename(args.input_csv, args.normalize_dir, "normalized_points", timestamp_str)
        steps.append({
            "name": "normalize",
            "cmd": [
                "python", "normalize.py",
                "--claims_csv", extract_output,
                "--debates_csv", args.input_csv,
                "--summaries_csv", summary_output,
                "--output_dir", args.normalize_dir,
                "--model_name", args.model_32b,
                *(["--add_timestamp"] if args.add_timestamp else []),
                *(["--resume"] if args.resume else []),
            ],
            "output": normalize_output
        })
    
    # Step 4: Cluster
    if not args.skip_cluster and "cluster" not in completed_steps:
        # Find latest normalize output
        normalize_output = find_latest_file(args.normalize_dir, f"{Path(args.input_csv).stem}_normalized_points")
        
        if not normalize_output:
            print("[ERROR] Cannot find normalize output for cluster step")
            return
        
        steps.append({
            "name": "cluster",
            "cmd": [
                "python", "cluster.py",
                "--input_csv", normalize_output,
                "--output_dir", args.cluster_dir,
                "--min_cluster_size", str(args.min_cluster_size),
                "--min_samples", str(args.min_samples),
                "--n_neighbors", str(args.n_neighbors),
                "--min_dist", str(args.min_dist),
                *(["--add_timestamp"] if args.add_timestamp else []),
            ],
            "output": None
        })
    
    # Step 5: Selection
    if not args.skip_selection and "selection" not in completed_steps:
        # Find latest cluster output
        cluster_output = find_latest_file(args.cluster_dir, f"{Path(args.input_csv).stem}_clustered_points")
        
        if not cluster_output:
            print("[ERROR] Cannot find cluster output for selection step")
            return
        
        steps.append({
            "name": "selection",
            "cmd": [
                "python", "selection.py",
                "--input_csv", cluster_output,
                "--output_dir", args.selection_dir,
                "--party", args.party,
                "--model_name", args.selection_model_name,
                "--threshold", str(args.selection_threshold),
                "--batch_size", str(args.selection_batch_size),
                *(["--keep_cluster_metadata"] if args.keep_cluster_metadata else []),
            ],
            "output": None
        })
    
    # Execute pipeline
    print(f"\n{'='*80}")
    print(f"BATCH PIPELINE: {args.party}")
    print(f"Input: {args.input_csv}")
    print(f"Steps to execute: {[s['name'] for s in steps]}")
    print(f"{'='*80}\n")
    
    for i, step in enumerate(steps, 1):
        print(f"\n[PROGRESS] Step {i}/{len(steps)}: {step['name'].upper()}")
        
        if not run_command(step["cmd"], f"{step['name'].capitalize()} step"):
            print(f"\n[ABORT] Pipeline stopped at {step['name']} step")
            return
        
        # Update checkpoint
        completed_steps.add(step["name"])
        with open(checkpoint_file, 'w') as f:
            json.dump({"completed_steps": list(completed_steps)}, f)
    
    # Success
    print(f"\n{'='*80}")
    print(f"[SUCCESS] Pipeline completed for {args.party}!")
    print(f"{'='*80}")
    
    # Cleanup checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


if __name__ == "__main__":
    main()
