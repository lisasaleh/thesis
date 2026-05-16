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
    
    def __init__(
        self,
        model_7b: str,
        model_32b: str,
        min_tokens: int = 100,
        data_dir: str = "/scratch-shared/lsaleh/debates/",
        backend: str = "local",
        api_base_url: str = "http://127.0.0.1:8000/v1",
        api_model_name: Optional[str] = None,
        api_key: str = "EMPTY",
        api_max_tokens: int = 1200,
        api_temperature: float = 0.0,
        api_timeout: float = 120.0,
        api_retries: int = 3,
        api_backoff: float = 2.0,
        cmp_ranks: Optional[str] = None,
        extract_max_new_tokens: int = 1200,
    ):
        self.model_7b = model_7b
        self.model_32b = model_32b
        self.min_tokens = min_tokens
        self.data_dir = data_dir
        self.backend = backend
        self.api_base_url = api_base_url
        self.api_model_name = api_model_name
        self.api_key = api_key
        self.api_max_tokens = api_max_tokens
        self.api_temperature = api_temperature
        self.api_timeout = api_timeout
        self.api_retries = api_retries
        self.api_backoff = api_backoff
        self.cmp_ranks = cmp_ranks
        self.extract_max_new_tokens = extract_max_new_tokens
        self.log_file = f"orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.results = {}

    def backend_args(self) -> List[str]:
        args = ["--backend", self.backend]
        if self.backend == "api":
            args.extend([
                "--api_base_url", self.api_base_url,
                "--api_key", self.api_key,
                "--api_max_tokens", str(self.api_max_tokens),
                "--api_temperature", str(self.api_temperature),
                "--api_timeout", str(self.api_timeout),
                "--api_retries", str(self.api_retries),
                "--api_backoff", str(self.api_backoff),
            ])
            if self.api_model_name:
                args.extend(["--api_model_name", self.api_model_name])
        return args
    
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
            "batch_party.py",
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
        if not os.path.exists(self.data_dir):
            self.log(f"  ✗ Missing data directory: {self.data_dir}")
            missing.append(self.data_dir)
        else:
            n_files = len(os.listdir(self.data_dir))
            self.log(f"  ✓ Data directory found: {self.data_dir} ({n_files} files)")
        
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
    
    def run_party_sequential(self, party: str, resume: bool = False, force: bool = False) -> bool:
        """
        Run pipeline for a single party (blocking).
        
        Returns True if successful, False otherwise.
        """
        self.log(f"\n{'='*80}")
        self.log(f"Starting processing for party: {party}")
        self.log(f"{'='*80}")
        
        cmd = [
            "python", "batch_party.py",
            "--party", party,
            "--model_7b", self.model_7b,
            "--model_32b", self.model_32b,
            "--min_tokens", str(self.min_tokens),
            "--data_dir", self.data_dir,
            "--extract_max_new_tokens", str(self.extract_max_new_tokens),
        ] + self.backend_args()
        
        if resume:
            cmd.append("--resume")
        if force:
            cmd.append("--force")
        if self.cmp_ranks:
            cmd.extend(["--cmp_ranks", self.cmp_ranks])
        
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
    
    def run_party_parallel_worker(self, party: str, resume: bool, force: bool, queue: Queue):
        """Worker function for parallel execution."""
        success = self.run_party_sequential(party, resume=resume, force=force)
        queue.put((party, success))
    
    def run_parties_sequential(self, parties: List[str], resume: bool = False, force: bool = False):
        """Run pipeline for multiple parties sequentially."""
        self.log(f"\nRunning {len(parties)} parties sequentially")
        
        for i, party in enumerate(parties, 1):
            self.log(f"\n[{i}/{len(parties)}] {party}")
            self.run_party_sequential(party, resume=resume, force=force)
    
    def run_parties_parallel(self, parties: List[str], resume: bool = False, force: bool = False, max_workers: int = 2):
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
                args=(party, resume, force, queue)
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
    parser.add_argument("--data_dir", type=str, default="/scratch-shared/lsaleh/debates",
                        help="Path to debates root directory (default: /scratch-shared/lsaleh/debates)")
    parser.add_argument("--min_tokens", type=int, default=30,
                        help="Minimum token threshold (lowered from 100 to include more debates)")
    parser.add_argument("--parties", type=str, default=None,
                        help="Comma-separated list of parties (default: all)")
    parser.add_argument("--mode", type=str, default="sequential",
                        choices=["sequential", "parallel"],
                        help="Execution mode")
    parser.add_argument("--max_workers", type=int, default=2,
                        help="Max parallel workers (only used in parallel mode)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints")
    parser.add_argument("--force", action="store_true",
                        help="Rerun completed batch_party stages")
    parser.add_argument("--cmp_ranks", type=str, default=None,
                        help="Comma-separated CMP ranks to process per party, e.g. '1' or '1,3,5'")
    parser.add_argument("--extract_max_new_tokens", type=int, default=int(os.environ.get("EXTRACT_MAX_NEW_TOKENS", "1200")),
                        help="Maximum new tokens for extraction JSON output")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate setup, don't run")
    parser.add_argument("--backend", choices=["local", "api"], default=os.environ.get("LLM_BACKEND", "local"),
                        help="Model backend: local transformers or OpenAI-compatible API")
    parser.add_argument("--api_base_url", type=str, default=os.environ.get("LLM_API_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api_model_name", type=str, default=os.environ.get("LLM_API_MODEL_NAME"))
    parser.add_argument("--api_key", type=str, default=os.environ.get("LLM_API_KEY", "EMPTY"))
    parser.add_argument("--api_max_tokens", type=int, default=int(os.environ.get("LLM_API_MAX_TOKENS", "1200")))
    parser.add_argument("--api_temperature", type=float, default=float(os.environ.get("LLM_API_TEMPERATURE", "0")))
    parser.add_argument("--api_timeout", type=float, default=float(os.environ.get("LLM_API_TIMEOUT", "120")))
    parser.add_argument("--api_retries", type=int, default=int(os.environ.get("LLM_API_RETRIES", "3")))
    parser.add_argument("--api_backoff", type=float, default=float(os.environ.get("LLM_API_BACKOFF", "2")))
    
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
    orchestrator = Orchestrator(
        args.model_7b,
        args.model_32b,
        args.min_tokens,
        args.data_dir,
        backend=args.backend,
        api_base_url=args.api_base_url,
        api_model_name=args.api_model_name,
        api_key=args.api_key,
        api_max_tokens=args.api_max_tokens,
        api_temperature=args.api_temperature,
        api_timeout=args.api_timeout,
        api_retries=args.api_retries,
        api_backoff=args.api_backoff,
        cmp_ranks=args.cmp_ranks,
        extract_max_new_tokens=args.extract_max_new_tokens,
    )
    
    # Validate setup
    if not orchestrator.validate_setup():
        sys.exit(1)
    
    # Run pipeline
    if args.mode == "sequential":
        orchestrator.run_parties_sequential(parties, resume=args.resume, force=args.force)
    else:
        orchestrator.run_parties_parallel(parties, resume=args.resume, force=args.force, max_workers=args.max_workers)
    
    # Print summary
    orchestrator.print_summary()


if __name__ == "__main__":
    main()
