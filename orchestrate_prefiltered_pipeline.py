import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PARTIES = [
    "50PLUS", "CDA", "CU", "D66", "GL", "PVDA",
    "PVDD", "PVV", "SGP", "SP", "VVD"
]


def read_csv_or_empty(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def run(cmd: list[str], description: str) -> bool:
    print("")
    print("=" * 80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}")
    print("=" * 80)
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[ERROR] {description} failed with exit code {result.returncode}")
        return False
    return True


def backend_args(args, model_name: str) -> list[str]:
    out = ["--backend", args.backend]
    if args.backend == "api":
        out.extend([
            "--api_base_url", args.api_base_url,
            "--api_model_name", args.api_model_name or model_name,
            "--api_key", args.api_key,
            "--api_max_tokens", str(args.api_max_tokens),
            "--api_temperature", str(args.api_temperature),
            "--api_timeout", str(args.api_timeout),
            "--api_retries", str(args.api_retries),
            "--api_backoff", str(args.api_backoff),
        ])
    return out


def party_cmd(args, party: str) -> list[str]:
    cmd = [
        "python", "batch_party_prefilter.py",
        "--party", party,
        "--model_7b", args.model_7b,
        "--model_32b", args.model_32b,
        "--min_tokens", str(args.min_tokens),
        "--extract_dir", args.extract_dir,
        "--prefilter_dir", args.prefilter_dir,
        "--summary_dir", args.summary_dir,
        "--normalize_dir", args.normalize_dir,
        "--log_dir", args.log_dir,
        "--debates_csv", args.debates_csv,
        "--cmp_manifest_csv", args.cmp_manifest_csv,
        "--data_dir", args.data_dir,
        "--extract_max_new_tokens", str(args.extract_max_new_tokens),
        "--extract_max_claims", str(args.extract_max_claims),
        "--chunk_max_words", str(args.chunk_max_words),
        "--prefilter_top_k", str(args.prefilter_top_k),
        "--roberta_batch_size", str(args.roberta_batch_size),
        "--roberta_device", args.roberta_device,
        "--roberta_model_name", args.roberta_model_name,
        "--roberta_tokenizer_name", args.roberta_tokenizer_name,
        "--force_stages", args.force_stages,
    ] + backend_args(args, args.model_7b)
    if args.cmp_ranks:
        cmd.extend(["--cmp_ranks", args.cmp_ranks])
    if args.resume:
        cmd.append("--resume")
    if args.force:
        cmd.append("--force")
    if args.final_cmp_recheck:
        cmd.append("--final_cmp_recheck")
    return cmd


def validation_old_cmd(args) -> list[str]:
    cmd = [
        "python", "batch_party.py",
        "--party", args.validation_party,
        "--model_7b", args.model_7b,
        "--model_32b", args.model_32b,
        "--min_tokens", str(args.min_tokens),
        "--extract_dir", args.validation_old_extract_dir,
        "--summary_dir", args.validation_old_summary_dir,
        "--normalize_dir", args.validation_old_normalize_dir,
        "--debates_csv", args.debates_csv,
        "--cmp_manifest_csv", args.cmp_manifest_csv,
        "--data_dir", args.data_dir,
        "--cmp_ranks", args.validation_cmp_ranks,
        "--extract_max_new_tokens", str(args.extract_max_new_tokens),
        "--extract_max_claims", str(args.extract_max_claims),
        "--chunk_max_words", str(args.chunk_max_words),
    ] + backend_args(args, args.model_7b)
    if args.resume:
        cmd.append("--resume")
    if args.force:
        cmd.append("--force")
    return cmd


def validation_new_cmd(args) -> list[str]:
    old_values = {
        "extract_dir": args.extract_dir,
        "prefilter_dir": args.prefilter_dir,
        "summary_dir": args.summary_dir,
        "normalize_dir": args.normalize_dir,
        "cmp_ranks": args.cmp_ranks,
    }
    args.extract_dir = args.validation_new_extract_dir
    args.prefilter_dir = args.validation_new_prefilter_dir
    args.summary_dir = args.validation_new_summary_dir
    args.normalize_dir = args.validation_new_normalize_dir
    args.cmp_ranks = args.validation_cmp_ranks
    cmd = party_cmd(args, args.validation_party)
    for key, value in old_values.items():
        setattr(args, key, value)
    return cmd


def compare_validation_outputs(args) -> None:
    rows = []
    ranks = [int(item.strip()) for item in args.validation_cmp_ranks.split(",") if item.strip()]
    for rank in ranks:
        old_norm = os.path.join(
            args.validation_old_normalize_dir,
            f"{args.validation_party}_cmp_{rank}_normalized.csv",
        )
        labeled = os.path.join(
            args.validation_new_prefilter_dir,
            f"{args.validation_party}_cmp_{rank}_claims_roberta_prefilter.csv",
        )
        old_df = read_csv_or_empty(old_norm)
        labeled_df = read_csv_or_empty(labeled)

        if old_df.empty or labeled_df.empty:
            rows.append({
                "party": args.validation_party,
                "cmp_rank": rank,
                "old_normalized_points": len(old_df),
                "old_points_found_in_raw_extraction": 0,
                "old_points_missing_from_new_raw_extraction": None,
                "old_points_lost_by_prefilter": None,
                "estimated_prefilter_recall": None,
                "note": "missing old normalized output or new prefilter output",
            })
            continue

        required_old_cols = {"document_id", "intervention_id", "claim_idx"}
        required_labeled_cols = {"document_id", "intervention_id", "claim_idx", "kept_by_prefilter"}
        missing_old = required_old_cols - set(old_df.columns)
        missing_labeled = required_labeled_cols - set(labeled_df.columns)
        if missing_old or missing_labeled:
            rows.append({
                "party": args.validation_party,
                "cmp_rank": rank,
                "old_normalized_points": len(old_df),
                "old_points_found_in_raw_extraction": 0,
                "old_points_missing_from_new_raw_extraction": None,
                "old_points_lost_by_prefilter": None,
                "estimated_prefilter_recall": None,
                "note": (
                    f"missing old columns={sorted(missing_old)}; "
                    f"missing labeled columns={sorted(missing_labeled)}"
                ),
            })
            continue

        old_keys = set(zip(old_df["document_id"], old_df["intervention_id"], old_df["claim_idx"]))
        labeled_lookup = labeled_df.set_index(["document_id", "intervention_id", "claim_idx"], drop=False)
        found = 0
        lost = 0
        missing_from_raw = 0
        for key in old_keys:
            if key not in labeled_lookup.index:
                missing_from_raw += 1
                continue
            found += 1
            row = labeled_lookup.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if str(row.get("kept_by_prefilter", "")).lower() != "true":
                lost += 1

        recall = None if found == 0 else (found - lost) / found
        rows.append({
            "party": args.validation_party,
            "cmp_rank": rank,
            "old_normalized_points": len(old_df),
            "old_points_found_in_raw_extraction": found,
            "old_points_missing_from_new_raw_extraction": missing_from_raw,
            "old_points_lost_by_prefilter": lost,
            "estimated_prefilter_recall": recall,
            "note": "",
        })

    report = pd.DataFrame(rows)
    Path(args.validation_report_csv).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.validation_report_csv, index=False)
    print("")
    print(report.to_string(index=False))
    print(f"Saved validation report to {args.validation_report_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Orchestrate the party-based prefiltered pipeline.")
    parser.add_argument("--model_7b", required=True)
    parser.add_argument("--model_32b", required=True)
    parser.add_argument("--parties", default=None)
    parser.add_argument("--cmp_ranks", default=None)
    parser.add_argument("--min_tokens", type=int, default=30)
    parser.add_argument("--extract_dir", default="/scratch-shared/lsaleh/extracted")
    parser.add_argument("--prefilter_dir", default="/scratch-shared/lsaleh/prefiltered")
    parser.add_argument("--summary_dir", default="/scratch-shared/lsaleh/summaries")
    parser.add_argument("--normalize_dir", default="/scratch-shared/lsaleh/normalized")
    parser.add_argument("--log_dir", default="/scratch-shared/lsaleh/prefiltered_logs")
    parser.add_argument("--debates_csv", default="outputs/debates.csv")
    parser.add_argument("--cmp_manifest_csv", default="outputs/cmp_manifest.csv")
    parser.add_argument("--data_dir", default="/scratch-shared/lsaleh/debates/")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force_stages", default="")
    parser.add_argument("--extract_max_new_tokens", type=int, default=1200)
    parser.add_argument("--extract_max_claims", type=int, default=20)
    parser.add_argument("--chunk_max_words", type=int, default=1000)
    parser.add_argument("--prefilter_top_k", type=int, default=3)
    parser.add_argument("--roberta_batch_size", type=int, default=16)
    parser.add_argument("--roberta_device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--roberta_model_name", default="manifesto-project/manifestoberta-xlm-roberta-56policy-topics-sentence-2024-1-1")
    parser.add_argument("--roberta_tokenizer_name", default="xlm-roberta-large")
    parser.add_argument("--final_cmp_recheck", action="store_true")
    parser.add_argument("--validation_mode", action="store_true")
    parser.add_argument("--validation_party", default="VVD")
    parser.add_argument("--validation_cmp_ranks", default="1")
    parser.add_argument("--validation_report_csv", default="/scratch-shared/lsaleh/validation/prefilter_recall_report.csv")
    parser.add_argument("--validation_old_extract_dir", default="/scratch-shared/lsaleh/validation/old/extracted")
    parser.add_argument("--validation_old_summary_dir", default="/scratch-shared/lsaleh/validation/old/summaries")
    parser.add_argument("--validation_old_normalize_dir", default="/scratch-shared/lsaleh/validation/old/normalized")
    parser.add_argument("--validation_new_extract_dir", default="/scratch-shared/lsaleh/validation/new/extracted")
    parser.add_argument("--validation_new_prefilter_dir", default="/scratch-shared/lsaleh/validation/new/prefiltered")
    parser.add_argument("--validation_new_summary_dir", default="/scratch-shared/lsaleh/validation/new/summaries")
    parser.add_argument("--validation_new_normalize_dir", default="/scratch-shared/lsaleh/validation/new/normalized")
    parser.add_argument("--backend", choices=["local", "api"], default=os.environ.get("LLM_BACKEND", "local"))
    parser.add_argument("--api_base_url", default=os.environ.get("LLM_API_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api_model_name", default=os.environ.get("LLM_API_MODEL_NAME"))
    parser.add_argument("--api_key", default=os.environ.get("LLM_API_KEY", "EMPTY"))
    parser.add_argument("--api_max_tokens", type=int, default=int(os.environ.get("LLM_API_MAX_TOKENS", "1200")))
    parser.add_argument("--api_temperature", type=float, default=float(os.environ.get("LLM_API_TEMPERATURE", "0")))
    parser.add_argument("--api_timeout", type=float, default=float(os.environ.get("LLM_API_TIMEOUT", "120")))
    parser.add_argument("--api_retries", type=int, default=int(os.environ.get("LLM_API_RETRIES", "3")))
    parser.add_argument("--api_backoff", type=float, default=float(os.environ.get("LLM_API_BACKOFF", "2")))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.validation_mode:
        old_ok = run(validation_old_cmd(args), "Validation old pipeline")
        new_ok = run(validation_new_cmd(args), "Validation new prefiltered pipeline")
        if old_ok and new_ok:
            compare_validation_outputs(args)
        sys.exit(0 if old_ok and new_ok else 1)

    parties = [item.strip() for item in args.parties.split(",")] if args.parties else PARTIES
    failed = []
    for party in parties:
        ok = run(party_cmd(args, party), f"Prefiltered pipeline for {party}")
        if not ok:
            failed.append(party)

    if failed:
        print(f"Failed parties: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
