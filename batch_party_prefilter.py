"""
Party-centric prefiltered pipeline:

debates -> raw claim extraction -> CMP RoBERTa top-k prefilter ->
summaries only for debates with retained claims -> normalization only for retained claims ->
optional CMP RoBERTa recheck on normalized points.

This is intentionally a wrapper around the existing stage scripts and helper
functions in batch_party.py. The old batch_party.py flow is left unchanged.
"""

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from batch_party import (
    ProcessingLogger,
    build_cmp_batch_input_csv,
    find_debates_for_themes,
    get_party_cmp_codes,
    load_cmp_manifest,
    run_command,
)


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


def mark_stage(logger: ProcessingLogger, completed: dict[int, list[str]], rank: int, stage: str) -> None:
    completed.setdefault(rank, [])
    if stage not in completed[rank]:
        completed[rank].append(stage)
        logger.save_checkpoint(completed)


def read_csv_or_empty(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def make_retained_inputs(
    batch_input_csv: str,
    labeled_claims_csv: str,
    retained_claim_chunks_csv: str,
    retained_debate_context_csv: str,
    retained_claims_csv: str,
) -> tuple[int, int, int]:
    claims_df = read_csv_or_empty(labeled_claims_csv)
    batch_df = read_csv_or_empty(batch_input_csv)

    if claims_df.empty:
        kept_claims = pd.DataFrame(columns=claims_df.columns)
        kept_claim_keys = set()
        kept_debate_ids = set()
    else:
        kept_claims = claims_df[claims_df["kept_by_prefilter"].astype(str).str.lower().eq("true")].copy()
        kept_claim_keys = set(zip(kept_claims["document_id"], kept_claims["intervention_id"]))
        kept_debate_ids = set(kept_claims["document_id"].dropna().tolist())

    if batch_df.empty or not kept_claim_keys:
        retained_claim_chunks = pd.DataFrame(columns=batch_df.columns)
    else:
        keys = list(zip(batch_df["document_id"], batch_df["intervention_id"]))
        retained_claim_chunks = batch_df[[key in kept_claim_keys for key in keys]].copy()

    if batch_df.empty or not kept_debate_ids:
        retained_debate_context = pd.DataFrame(columns=batch_df.columns)
    else:
        retained_debate_context = batch_df[batch_df["document_id"].isin(kept_debate_ids)].copy()

    Path(retained_claims_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(retained_claim_chunks_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(retained_debate_context_csv).parent.mkdir(parents=True, exist_ok=True)
    kept_claims.to_csv(retained_claims_csv, index=False)
    retained_claim_chunks.to_csv(retained_claim_chunks_csv, index=False)
    retained_debate_context.to_csv(retained_debate_context_csv, index=False)
    return len(kept_claims), len(retained_claim_chunks), len(retained_debate_context)


def force_stage(args, stage: str) -> bool:
    return args.force or stage in args.force_stages_set


def combine_summary_files(
    temp_summary_dir: str,
    summary_dir: str,
    combined_summary_output: str,
    party: str,
    rank: int,
) -> Optional[str]:
    moved_files = []
    for summary_file in os.listdir(temp_summary_dir):
        if not summary_file.endswith("_summary.csv") or summary_file == "combined_summary.csv":
            continue
        doc_id = summary_file.replace("_summary.csv", "")
        src = os.path.join(temp_summary_dir, summary_file)
        dst = os.path.join(summary_dir, f"{party}_cmp_{rank}_{doc_id}_prefiltered_summarized.csv")
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
        moved_files.append(dst)

    if not moved_files:
        return None

    combined = pd.concat([pd.read_csv(path) for path in moved_files], ignore_index=True)
    Path(combined_summary_output).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_summary_output, index=False)
    return combined_summary_output


def process_party_prefiltered(args) -> None:
    args.force_stages_set = {
        item.strip()
        for item in str(args.force_stages or "").split(",")
        if item.strip()
    }
    logger = ProcessingLogger(args.party, log_dir=args.log_dir)
    logger.checkpoint_file = os.path.join(args.log_dir, f".checkpoint_{args.party}_prefiltered.json")
    logger.log(f"Starting PREFILTERED pipeline for party: {args.party}")
    logger.log(f"Backend: {args.backend}")
    logger.log(f"RoBERTa top_k prefilter: {args.prefilter_top_k}")
    if args.force_stages_set:
        logger.log(f"Force stages: {', '.join(sorted(args.force_stages_set))}")

    for directory in [args.extract_dir, args.prefilter_dir, args.summary_dir, args.normalize_dir]:
        os.makedirs(directory, exist_ok=True)

    completed = {} if args.force else logger.load_checkpoint()
    cmp_manifest = load_cmp_manifest(args.cmp_manifest_csv)
    cmp_codes = get_party_cmp_codes(cmp_manifest, args.party)
    if args.cmp_ranks:
        wanted = {int(item.strip()) for item in args.cmp_ranks.split(",") if item.strip()}
        cmp_codes = [item for item in cmp_codes if item["rank"] in wanted]

    if not cmp_codes:
        raise ValueError(f"No CMP codes found for party={args.party!r}")

    stats = {
        "processed": 0,
        "completed": 0,
        "failed": 0,
        "raw_claims": 0,
        "kept_claims": 0,
        "retained_claim_chunks": 0,
        "retained_context_chunks": 0,
    }

    for cmp_info in cmp_codes:
        rank = int(cmp_info["rank"])
        code = str(cmp_info["code"])
        completed.setdefault(rank, [])

        logger.log("")
        logger.log("=" * 80)
        logger.log(f"Processing CMP rank {rank}: code={code} title={cmp_info['title']}")
        logger.log("=" * 80)

        batch_input_dir = os.path.join(args.extract_dir, "batch_inputs")
        temp_batch_input = os.path.join(batch_input_dir, f"{args.party}_cmp_{rank}_input.csv")
        retained_claim_chunks_csv = os.path.join(batch_input_dir, f"{args.party}_cmp_{rank}_retained_claim_chunks.csv")
        retained_debate_context_csv = os.path.join(batch_input_dir, f"{args.party}_cmp_{rank}_retained_debate_context.csv")
        extracted_final = os.path.join(args.extract_dir, f"{args.party}_cmp_{rank}_claims.csv")
        labeled_claims_csv = os.path.join(args.prefilter_dir, f"{args.party}_cmp_{rank}_claims_roberta_prefilter.csv")
        retained_claims_csv = os.path.join(args.prefilter_dir, f"{args.party}_cmp_{rank}_claims_kept.csv")
        combined_summary_output = os.path.join(
            args.summary_dir,
            f"{args.party}_cmp_{rank}_prefiltered_combined_summary_for_normalize.csv",
        )
        normalized_final = os.path.join(args.normalize_dir, f"{args.party}_cmp_{rank}_prefiltered_normalized.csv")
        final_recheck_csv = os.path.join(args.prefilter_dir, f"{args.party}_cmp_{rank}_normalized_roberta_recheck.csv")

        if (
            not args.force
            and os.path.exists(extracted_final)
            and os.path.exists(labeled_claims_csv)
            and os.path.exists(retained_claims_csv)
            and os.path.exists(retained_claim_chunks_csv)
            and os.path.exists(retained_debate_context_csv)
            and os.path.exists(combined_summary_output)
            and os.path.exists(normalized_final)
            and (not args.final_cmp_recheck or os.path.exists(final_recheck_csv))
        ):
            logger.log(f"[SKIPPED] Rank {rank} already has all final outputs")
            stats["completed"] += 1
            continue

        debate_ids = find_debates_for_themes(cmp_info["theme_ids"], args.debates_csv)
        if not debate_ids:
            logger.log(f"No debates found for CMP rank {rank}", level="WARN")
            stats["failed"] += 1
            continue

        Path(batch_input_dir).mkdir(parents=True, exist_ok=True)
        if force_stage(args, "batch_input") or not os.path.exists(temp_batch_input):
            num_included, num_skipped = build_cmp_batch_input_csv(
                args.party,
                cmp_info,
                debate_ids,
                args.debates_csv,
                args.data_dir,
                temp_batch_input,
                args.min_tokens,
                args.chunk_max_words,
                logger,
            )
        else:
            batch_df = pd.read_csv(temp_batch_input)
            num_included, num_skipped = len(batch_df), 0
            logger.log(f"[SKIPPED] Batch input exists: {temp_batch_input} ({num_included} rows)")

        if num_included == 0:
            logger.log(f"No valid input rows for CMP rank {rank}", level="WARN")
            stats["failed"] += 1
            continue

        stats["processed"] += 1

        if not force_stage(args, "extract") and os.path.exists(extracted_final):
            logger.log(f"[SKIPPED] Extraction output exists: {extracted_final}")
            mark_stage(logger, completed, rank, "extract")
        else:
            temp_extract_dir = os.path.join(".tmp_batch_prefilter", f"extract_cmp_{rank}")
            os.makedirs(temp_extract_dir, exist_ok=True)
            extract_cmd = [
                "python", "extract.py",
                "--input_csv", temp_batch_input,
                "--output_dir", temp_extract_dir,
                "--party", args.party,
                "--model_name", args.model_7b,
                "--target_party", args.party,
                "--extract_max_new_tokens", str(args.extract_max_new_tokens),
                "--extract_max_claims", str(args.extract_max_claims),
            ] + backend_args(args, args.model_7b)
            if args.resume:
                extract_cmd.append("--resume")
            ok = run_command(extract_cmd, f"Raw extraction for {args.party} CMP {rank}", fatal=False)
            produced = os.path.join(temp_extract_dir, f"{args.party}_claims.csv")
            if not ok or not os.path.exists(produced):
                logger.log(f"Extraction failed or produced no claims file for rank {rank}", level="ERROR")
                stats["failed"] += 1
                continue
            shutil.move(produced, extracted_final)
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
            mark_stage(logger, completed, rank, "extract")

        raw_claims_df = read_csv_or_empty(extracted_final)
        stats["raw_claims"] += len(raw_claims_df)

        if not force_stage(args, "prefilter") and os.path.exists(labeled_claims_csv):
            logger.log(f"[SKIPPED] RoBERTa prefilter output exists: {labeled_claims_csv}")
            mark_stage(logger, completed, rank, "prefilter")
        else:
            prefilter_cmd = [
                "python", "cmp_roberta_prefilter.py",
                "--input_csv", extracted_final,
                "--output_csv", labeled_claims_csv,
                "--manifest_file", args.cmp_manifest_csv,
                "--party", args.party,
                "--cmp_rank", str(rank),
                "--text_col", "quote",
                "--top_k", str(args.prefilter_top_k),
                "--batch_size", str(args.roberta_batch_size),
                "--model_name", args.roberta_model_name,
                "--tokenizer_name", args.roberta_tokenizer_name,
            ]
            if force_stage(args, "prefilter"):
                prefilter_cmd.append("--force")
            ok = run_command(prefilter_cmd, f"RoBERTa prefilter for {args.party} CMP {rank}", fatal=False)
            if not ok or not os.path.exists(labeled_claims_csv):
                logger.log(f"RoBERTa prefilter failed for rank {rank}", level="ERROR")
                stats["failed"] += 1
                continue
            mark_stage(logger, completed, rank, "prefilter")

        if (
            force_stage(args, "retained_inputs")
            or not (
                os.path.exists(retained_claims_csv)
                and os.path.exists(retained_claim_chunks_csv)
                and os.path.exists(retained_debate_context_csv)
            )
        ):
            kept_claims, retained_claim_chunks, retained_context_chunks = make_retained_inputs(
                batch_input_csv=temp_batch_input,
                labeled_claims_csv=labeled_claims_csv,
                retained_claim_chunks_csv=retained_claim_chunks_csv,
                retained_debate_context_csv=retained_debate_context_csv,
                retained_claims_csv=retained_claims_csv,
            )
            logger.log(
                "Retained by prefilter: "
                f"claims={kept_claims}, claim_chunks={retained_claim_chunks}, "
                f"context_chunks={retained_context_chunks}"
            )
        else:
            kept_claims = len(read_csv_or_empty(retained_claims_csv))
            retained_claim_chunks = len(read_csv_or_empty(retained_claim_chunks_csv))
            retained_context_chunks = len(read_csv_or_empty(retained_debate_context_csv))
            logger.log(
                "[SKIPPED] Retained inputs exist: "
                f"claims={kept_claims}, claim_chunks={retained_claim_chunks}, "
                f"context_chunks={retained_context_chunks}"
            )
        stats["kept_claims"] += kept_claims
        stats["retained_claim_chunks"] += retained_claim_chunks
        stats["retained_context_chunks"] += retained_context_chunks
        mark_stage(logger, completed, rank, "retained_inputs")

        if kept_claims == 0 or retained_claim_chunks == 0 or retained_context_chunks == 0:
            logger.log(f"No retained claims/context for CMP rank {rank}; writing empty downstream outputs", level="WARN")
            Path(combined_summary_output).parent.mkdir(parents=True, exist_ok=True)
            Path(normalized_final).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=[
                "document_id",
                "intervention_id",
                "party",
                "speaker",
                "speaker_label",
                "speech",
                "running_summary_after",
            ]).to_csv(combined_summary_output, index=False)
            pd.DataFrame(columns=[
                "document_id",
                "intervention_id",
                "party",
                "speaker",
                "speaker_label",
                "claim_idx",
                "quote",
                "point",
            ]).to_csv(normalized_final, index=False)
            mark_stage(logger, completed, rank, "summarize")
            mark_stage(logger, completed, rank, "normalize")
            stats["completed"] += 1
            continue

        if not force_stage(args, "summarize") and os.path.exists(combined_summary_output):
            logger.log(f"[SKIPPED] Summary output exists: {combined_summary_output}")
            mark_stage(logger, completed, rank, "summarize")
        else:
            temp_summary_dir = os.path.join(".tmp_batch_prefilter", f"summary_cmp_{rank}")
            os.makedirs(temp_summary_dir, exist_ok=True)
            summary_cmd = [
                "python", "incr_summary.py",
                "--input_csv", retained_debate_context_csv,
                "--output_dir", temp_summary_dir,
                "--model_name", args.model_7b,
            ] + backend_args(args, args.model_7b)
            if args.resume:
                summary_cmd.append("--resume")
            ok = run_command(summary_cmd, f"Summarization for retained {args.party} CMP {rank}", fatal=False)
            if not ok:
                logger.log(f"Summarization failed for rank {rank}", level="ERROR")
                stats["failed"] += 1
                continue
            summary_output = combine_summary_files(
                temp_summary_dir,
                args.summary_dir,
                combined_summary_output,
                args.party,
                rank,
            )
            shutil.rmtree(temp_summary_dir, ignore_errors=True)
            if not summary_output:
                logger.log(f"No summary files produced for rank {rank}", level="ERROR")
                stats["failed"] += 1
                continue
            mark_stage(logger, completed, rank, "summarize")

        if not force_stage(args, "normalize") and os.path.exists(normalized_final):
            logger.log(f"[SKIPPED] Normalization output exists: {normalized_final}")
            mark_stage(logger, completed, rank, "normalize")
        else:
            temp_normalize_dir = os.path.join(".tmp_batch_prefilter", f"normalize_cmp_{rank}")
            os.makedirs(temp_normalize_dir, exist_ok=True)
            normalize_cmd = [
                "python", "normalize.py",
                "--claims_csv", retained_claims_csv,
                "--debates_csv", retained_debate_context_csv,
                "--summaries_csv", combined_summary_output,
                "--output_dir", temp_normalize_dir,
                "--party", args.party,
                "--model_name", args.model_32b,
            ] + backend_args(args, args.model_32b)
            if args.resume:
                normalize_cmd.append("--resume")
            ok = run_command(normalize_cmd, f"Normalization for retained {args.party} CMP {rank}", fatal=False)
            produced = os.path.join(temp_normalize_dir, f"{args.party}_normalized.csv")
            if not ok or not os.path.exists(produced):
                logger.log(f"Normalization failed or produced no point file for rank {rank}", level="ERROR")
                stats["failed"] += 1
                continue
            shutil.move(produced, normalized_final)
            shutil.rmtree(temp_normalize_dir, ignore_errors=True)
            mark_stage(logger, completed, rank, "normalize")

        if args.final_cmp_recheck:
            if not force_stage(args, "final_cmp_recheck") and os.path.exists(final_recheck_csv):
                logger.log(f"[SKIPPED] Final CMP recheck exists: {final_recheck_csv}")
            else:
                recheck_cmd = [
                    "python", "cmp_roberta_prefilter.py",
                    "--input_csv", normalized_final,
                    "--output_csv", final_recheck_csv,
                    "--manifest_file", args.cmp_manifest_csv,
                    "--party", args.party,
                    "--cmp_rank", str(rank),
                    "--text_col", "point",
                    "--top_k", str(args.prefilter_top_k),
                    "--batch_size", str(args.roberta_batch_size),
                    "--model_name", args.roberta_model_name,
                    "--tokenizer_name", args.roberta_tokenizer_name,
                ]
                if force_stage(args, "final_cmp_recheck"):
                    recheck_cmd.append("--force")
                ok = run_command(recheck_cmd, f"Final RoBERTa recheck for {args.party} CMP {rank}", fatal=False)
                if not ok:
                    logger.log(f"Final RoBERTa recheck failed for rank {rank}", level="WARN")
                else:
                    mark_stage(logger, completed, rank, "final_cmp_recheck")

        logger.log(f"SUCCESS: CMP rank {rank} prefiltered pipeline completed")
        stats["completed"] += 1

    logger.log("")
    logger.log("=" * 80)
    logger.log(f"PREFILTERED SUMMARY FOR {args.party}")
    logger.log("=" * 80)
    for key, value in stats.items():
        logger.log(f"{key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run one party through the prefiltered CMP pipeline.")
    parser.add_argument("--party", required=True)
    parser.add_argument("--model_7b", required=True)
    parser.add_argument("--model_32b", required=True)
    parser.add_argument("--min_tokens", type=int, default=30)
    parser.add_argument("--extract_dir", default="/scratch-shared/lsaleh/extracted")
    parser.add_argument("--prefilter_dir", default="/scratch-shared/lsaleh/prefiltered")
    parser.add_argument("--summary_dir", default="/scratch-shared/lsaleh/summaries")
    parser.add_argument("--normalize_dir", default="/scratch-shared/lsaleh/normalized")
    parser.add_argument("--log_dir", default="outputs/logs")
    parser.add_argument("--debates_csv", default="outputs/debates.csv")
    parser.add_argument("--cmp_manifest_csv", default="outputs/cmp_manifest.csv")
    parser.add_argument("--data_dir", default="/scratch-shared/lsaleh/debates/")
    parser.add_argument("--cmp_ranks", default=None)
    parser.add_argument("--resume", action="store_true", help="Accepted for compatibility; outputs are always checked.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--force_stages",
        default="",
        help=(
            "Comma-separated stages to rerun without forcing the whole pipeline. "
            "Valid names include batch_input, extract, prefilter, retained_inputs, "
            "summarize, normalize, final_cmp_recheck."
        ),
    )
    parser.add_argument("--extract_max_new_tokens", type=int, default=1200)
    parser.add_argument("--extract_max_claims", type=int, default=20)
    parser.add_argument("--chunk_max_words", type=int, default=1000)
    parser.add_argument("--prefilter_top_k", type=int, default=3)
    parser.add_argument("--roberta_batch_size", type=int, default=16)
    parser.add_argument("--roberta_model_name", default="manifesto-project/manifestoberta-xlm-roberta-56policy-topics-sentence-2024-1-1")
    parser.add_argument("--roberta_tokenizer_name", default="xlm-roberta-large")
    parser.add_argument("--final_cmp_recheck", action="store_true")
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
    process_party_prefiltered(args)


if __name__ == "__main__":
    main()
