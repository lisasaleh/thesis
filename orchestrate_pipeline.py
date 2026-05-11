"""
Multi-party orchestrator for batch_all_party_centric.py

Runs the party-centric pipeline for multiple parties, either sequentially
or in parallel. Also provides utilities for validation and testing.

Usage:
    # Run all parties sequentially
    python orchestrate_pipeline.py --mode sequential --model_7b Qwen/Qwen2.5-7B-Instruct \\
        --model_32b /path/to/Qwen2.5-32B-Instruct
    
    # Run specific parties in parallel (use with caution - memory intensive!)
    python orchestrate_pipeline.py --parties VVD,PvdA,D66 --mode parallel \\
        --model_7b Qwen/Qwen2.5-7B-Instruct --model_32b /path/to/Qwen2.5-32B-Instruct
    
    # Resume all parties from checkpoints
    python orchestrate_pipeline.py --mode sequential --resume \\
        --model_7b Qwen/Qwen2.5-7B-Instruct --model_32b /path/to/Qwen2.5-32B-Instruct
    
    # Validate setup before running
    python orchestrate_pipeline.py --validate-only
"""

import os
import subprocess
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd
from multiprocessing import Process, Queue


PARTIES = [
    "50PLUS", "CDA", "CU", "D66", "GL", "PVDA",
    "PVDD", "PVV", "SGP", "SP", "VVD"
]


class Orchestrator:
    """Orchestrate party-centric pipeline across multiple parties."""
    
    def __init__(self, model_7b: str, model_32b: str, min_tokens: int = 300):
        self.model_7b = model_7b
        self.model_32b = model_32b
        self.min_tokens = min_tokens
        self.log_file = f"orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.results = {}
    
    def log(self, message: str):
        """Log to file and stdout."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + "\n")
    
    def validate_setup(self) -> bool:
        """Validate that all required files and directories exist."""
        self.log("Validating pipeline setup...")
        
        required_files = [
            "outputs/cmp_manifest.csv",
            "outputs/debates.csv",
            "batch_all_party_centric.py",
            "extract.py",
            "incr_summary.py",
            "normalize.py"
        ]
        
        missing = []
        for file in required_files:
            if not os.path.exists(file):
                missing.append(file)
                self.log(f"  ✗ Missing: {file}")
            else:
                self.log(f"  ✓ Found: {file}")
        
        # Check for data directory
        if not os.path.exists("data/2012"):
            self.log("  ✗ Missing data directory: data/2012")
            missing.append("data/2012")
        else:
            n_files = len(os.listdir("data/2012"))
            self.log(f"  ✓ Data directory found: data/2012 ({n_files} files)")
        
        # Check models are accessible
        self.log(f"\n  Model 7B: {self.model_7b}")
        self.log(f"  Model 32B: {self.model_32b}")
        
        # Load and validate CMP manifest
        try:
            cmp_df = pd.read_csv("outputs/cmp_manifest.csv")
            available_parties = cmp_df["party"].unique().tolist()
            self.log(f"\n  Available parties in CMP manifest: {available_parties}")
        except Exception as e:
            self.log(f"  ✗ Error reading CMP manifest: {e}")
            return False
        
        if missing:
            self.log(f"\nValidation FAILED: {len(missing)} required files/directories missing")
            return False
        else:
            self.log("\nValidation PASSED: All required files present")
            return True
    
    def run_party_sequential(self, party: str, resume: bool = False) -> bool:
        """
        Run pipeline for a single party (blocking).
        
        Returns True if successful, False otherwise.
        """
        self.log(f"\n{'='*80}")
        self.log(f"Starting processing for party: {party}")
        self.log(f"{'='*80}")
        
        cmd = [
            "python", "batch_all_party_centric.py",
            "--party", party,
            "--model_7b", self.model_7b,
            "--model_32b", self.model_32b,
            "--min_tokens", str(self.min_tokens)
        ]
        
        if resume:
            cmd.append("--resume")
        
        try:
            result = subprocess.run(cmd, check=False, capture_output=False)
            success = result.returncode == 0
            
            if success:
                self.log(f"✓ {party} completed successfully")
            else:
                self.log(f"✗ {party} failed with exit code {result.returncode}")
            
            self.results[party] = {
                "success": success,
                "exit_code": result.returncode
            }
            
            return success
            
        except Exception as e:
            self.log(f"✗ {party} encountered exception: {e}")
            self.results[party] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def run_party_parallel_worker(self, party: str, resume: bool, queue: Queue):
        """Worker function for parallel execution."""
        success = self.run_party_sequential(party, resume=resume)
        queue.put((party, success))
    
    def run_parties_sequential(self, parties: List[str], resume: bool = False):
        """Run pipeline for multiple parties sequentially."""
        self.log(f"\nRunning {len(parties)} parties sequentially")
        
        for i, party in enumerate(parties, 1):
            self.log(f"\n[{i}/{len(parties)}] {party}")
            self.run_party_sequential(party, resume=resume)
    
    def run_parties_parallel(self, parties: List[str], resume: bool = False, max_workers: int = 2):
        """
        Run pipeline for multiple parties in parallel.
        
        WARNING: This is memory-intensive. Only use with small number of parties.
        """
        self.log(f"\nRunning {len(parties)} parties in parallel (max_workers={max_workers})")
        
        queue = Queue()
        processes = []
        
        for party in parties:
            # Wait if we've reached max workers
            while len(processes) >= max_workers:
                try:
                    completed_party, success = queue.get(timeout=1)
                    self.log(f"Worker completed: {completed_party} (success={success})")
                    # Remove completed process
                    processes = [p for p in processes if p.is_alive()]
                except:
                    pass
            
            # Start new process
            p = Process(
                target=self.run_party_parallel_worker,
                args=(party, resume, queue)
            )
            p.start()
            processes.append(p)
            self.log(f"Started worker for {party}")
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Collect remaining results
        while not queue.empty():
            party, success = queue.get()
            self.log(f"Worker completed: {party} (success={success})")
    
    def print_summary(self):
        """Print summary of all processing."""
        self.log(f"\n{'='*80}")
        self.log("PROCESSING SUMMARY")
        self.log(f"{'='*80}")
        
        successful = [p for p, r in self.results.items() if r.get("success")]
        failed = [p for p, r in self.results.items() if not r.get("success")]
        
        self.log(f"Total parties processed: {len(self.results)}")
        self.log(f"Successful: {len(successful)}")
        self.log(f"Failed: {len(failed)}")
        
        if successful:
            self.log(f"\n✓ Successful: {', '.join(successful)}")
        
        if failed:
            self.log(f"\n✗ Failed: {', '.join(failed)}")
            for party in failed:
                error_info = self.results[party]
                if "error" in error_info:
                    self.log(f"    {party}: {error_info['error']}")
                else:
                    self.log(f"    {party}: exit code {error_info.get('exit_code', 'unknown')}")
        
        self.log(f"\nResults saved to: {self.log_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrate party-centric pipeline across multiple parties"
    )
    
    parser.add_argument("--model_7b", type=str,
                        help="7B model name")
    parser.add_argument("--model_32b", type=str,
                        help="32B model name")
    parser.add_argument("--min_tokens", type=int, default=300,
                        help="Minimum token threshold")
    parser.add_argument("--parties", type=str, default=None,
                        help="Comma-separated list of parties (default: all)")
    parser.add_argument("--mode", type=str, default="sequential",
                        choices=["sequential", "parallel"],
                        help="Execution mode")
    parser.add_argument("--max_workers", type=int, default=2,
                        help="Max parallel workers (only used in parallel mode)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate setup, don't run")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Handle validate-only mode
    if args.validate_only:
        orchestrator = Orchestrator("dummy", "dummy")
        success = orchestrator.validate_setup()
        sys.exit(0 if success else 1)
    
    # Require models for actual execution
    if not args.model_7b or not args.model_32b:
        print("ERROR: --model_7b and --model_32b are required (unless --validate-only)")
        sys.exit(1)
    
    # Parse parties
    if args.parties:
        parties = [p.strip() for p in args.parties.split(",")]
    else:
        parties = PARTIES
    
    # Create orchestrator
    orchestrator = Orchestrator(args.model_7b, args.model_32b, args.min_tokens)
    
    # Validate setup
    if not orchestrator.validate_setup():
        sys.exit(1)
    
    # Run pipeline
    if args.mode == "sequential":
        orchestrator.run_parties_sequential(parties, resume=args.resume)
    else:
        orchestrator.run_parties_parallel(parties, resume=args.resume, max_workers=args.max_workers)
    
    # Print summary
    orchestrator.print_summary()


if __name__ == "__main__":
    main()
