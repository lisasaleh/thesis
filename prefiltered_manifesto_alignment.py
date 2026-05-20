import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent.parent if SCRIPT_PATH.parent.name == "scripts_analyse" else SCRIPT_PATH.parent
DEFAULT_PREFILTER_ROOT = Path("/scratch_shared/lsaleh/prefilter_pipeline")
ALT_PREFILTER_ROOT = Path("/scratch-shared/lsaleh/prefilter_pipeline")
DEFAULT_MANIFESTO_INDEX = ROOT / "outputs/embeddings/manifesto_sbert_embedding_index.csv"
DEFAULT_MANIFESTO_EMB = ROOT / "outputs/embeddings/manifesto_sbert_embeddings.npy"
ALT_MANIFESTO_INDEX = ROOT / "outputs/manifesto/embeddings/manifesto_sbert_embedding_index.csv"
ALT_MANIFESTO_EMB = ROOT / "outputs/manifesto/embeddings/manifesto_sbert_embeddings.npy"
DEFAULT_CMP_MANIFEST = ROOT / "outputs/cmp_manifest.csv"
DEFAULT_DEBATES_CSV = ROOT / "outputs/debates.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/analysis/prefiltered_manifesto_alignment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze debate-claim alignment to manifesto anchors over date_count."
    )
    parser.add_argument(
        "--parties",
        default=None,
        help="Comma-separated parties. If omitted, load all party embedding indexes found under prefilter_root.",
    )
    parser.add_argument(
        "--prefilter_root",
        type=Path,
        default=DEFAULT_PREFILTER_ROOT,
        help="Root containing {party}/embeddings outputs from embed_prefiltered_claims.py.",
    )
    parser.add_argument("--manifesto_index", type=Path, default=DEFAULT_MANIFESTO_INDEX)
    parser.add_argument("--manifesto_embeddings", type=Path, default=DEFAULT_MANIFESTO_EMB)
    parser.add_argument(
        "--cmp_manifest",
        type=Path,
        default=DEFAULT_CMP_MANIFEST,
        help="Party CMP-rank manifest used when claim indexes only have cmp_rank.",
    )
    parser.add_argument("--debates_csv", type=Path, default=DEFAULT_DEBATES_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min_claims_per_date", type=int, default=5)
    parser.add_argument("--dominance_threshold", type=float, default=0.25)
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=13)
    parser.add_argument("--min_bootstrap_claims", type=int, default=10)
    parser.add_argument("--mmd_sample_size", type=int, default=500)
    parser.add_argument("--min_claims_for_smoothing", type=int, default=30)
    parser.add_argument("--lowess_frac", type=float, default=0.25)
    parser.add_argument("--n_temporal_bins", type=int, default=20)
    parser.add_argument("--cabinet_start_date", default="2012-11-05")
    parser.add_argument("--pre_election_date", default="2016-09-15")
    parser.add_argument("--early_period_quantile", type=float, default=0.33)
    parser.add_argument("--election_period_quantile", type=float, default=0.80)
    parser.add_argument(
        "--max_plot_series",
        type=int,
        default=80,
        help="Plot only the largest party-CMP series to avoid unreadable plot folders.",
    )
    return parser.parse_args()


def resolve_prefilter_root(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_PREFILTER_ROOT and ALT_PREFILTER_ROOT.exists():
        print(f"[WARN] {DEFAULT_PREFILTER_ROOT} not found; using {ALT_PREFILTER_ROOT}")
        return ALT_PREFILTER_ROOT
    return path


def resolve_manifesto_paths(index_path: Path, emb_path: Path) -> tuple[Path, Path]:
    if index_path.exists() and emb_path.exists():
        return index_path, emb_path
    if index_path == DEFAULT_MANIFESTO_INDEX and emb_path == DEFAULT_MANIFESTO_EMB:
        if ALT_MANIFESTO_INDEX.exists() and ALT_MANIFESTO_EMB.exists():
            print(f"[WARN] Default manifesto embeddings not found; using {ALT_MANIFESTO_INDEX.parent}")
            return ALT_MANIFESTO_INDEX, ALT_MANIFESTO_EMB
    return index_path, emb_path


def parse_parties(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def normalize_vec(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def read_party_embeddings(prefilter_root: Path, parties: list[str] | None) -> tuple[pd.DataFrame, np.ndarray]:
    if parties is None:
        index_paths = sorted(prefilter_root.glob("*/embeddings/*_prefiltered_claim_sbert_embedding_index.csv"))
    else:
        index_paths = [
            prefilter_root / party / "embeddings" / f"{party}_prefiltered_claim_sbert_embedding_index.csv"
            for party in parties
        ]

    frames = []
    vectors = []
    offset = 0
    missing = []

    for index_path in index_paths:
        if not index_path.exists():
            missing.append(str(index_path))
            continue

        party = index_path.name.replace("_prefiltered_claim_sbert_embedding_index.csv", "")
        emb_path = index_path.with_name(f"{party}_prefiltered_claim_sbert_embeddings.npy")
        if not emb_path.exists():
            missing.append(str(emb_path))
            continue

        df = pd.read_csv(index_path)
        emb = np.load(emb_path)
        if len(df) != len(emb):
            raise ValueError(f"Index and embeddings do not align for {party}: {len(df)} rows vs {len(emb)} vectors")

        df = df.copy()
        df["local_embedding_id"] = df["embedding_id"]
        df["embedding_id"] = np.arange(offset, offset + len(df))
        df["embedding_file"] = str(emb_path)
        frames.append(df)
        vectors.append(emb)
        offset += len(df)

    if missing:
        print("[WARN] Missing embedding inputs:")
        for item in missing:
            print(f"  {item}")

    if not frames:
        raise FileNotFoundError(f"No party claim embeddings found under {prefilter_root}")

    return pd.concat(frames, ignore_index=True), normalize_rows(np.vstack(vectors))


def load_manifesto(index_path: Path, emb_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    if not index_path.exists():
        raise FileNotFoundError(f"Manifesto index not found: {index_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Manifesto embeddings not found: {emb_path}")

    df = pd.read_csv(index_path)
    emb = np.load(emb_path)
    if len(df) != len(emb):
        raise ValueError(f"Manifesto index and embeddings do not align: {len(df)} rows vs {len(emb)} vectors")

    df = df.copy()
    df["cmp_code"] = pd.to_numeric(df["cmp_code"], errors="coerce").astype("Int64")
    df["party_norm"] = df["party"].astype(str).str.upper()
    return df, normalize_rows(emb)


def load_cmp_rank_map(path: Path) -> dict[tuple[str, int], int]:
    if not path.exists():
        print(f"[WARN] CMP manifest not found: {path}")
        return {}

    manifest = pd.read_csv(path)
    if "party" not in manifest.columns:
        print(f"[WARN] CMP manifest missing party column: {path}")
        return {}

    mapping = {}
    for _, row in manifest.iterrows():
        party = str(row["party"]).upper()
        for rank in range(1, 11):
            col = f"code_{rank}"
            if col not in manifest.columns or pd.isna(row.get(col)):
                continue
            mapping[(party, rank)] = int(row[col])
    return mapping


def prepare_claims(df: pd.DataFrame, cmp_rank_map: dict[tuple[str, int], int]) -> pd.DataFrame:
    required = {"document_id", "datecount", "party", "claim_idx", "point", "embedding_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Claim embedding index missing required columns: {missing}")

    out = df.copy()
    out["date_count"] = pd.to_numeric(out["datecount"], errors="coerce")
    out["party_norm"] = out["party"].astype(str).str.upper()

    if "cmp_code" in out.columns:
        out["cmp_code"] = pd.to_numeric(out["cmp_code"], errors="coerce").astype("Int64")
    elif "target_cmp_code" in out.columns:
        out["cmp_code"] = pd.to_numeric(out["target_cmp_code"], errors="coerce").astype("Int64")
    elif "predicted_cmp_code" in out.columns:
        out["cmp_code"] = pd.to_numeric(out["predicted_cmp_code"], errors="coerce").astype("Int64")
    elif "cmp_rank" in out.columns and cmp_rank_map:
        out["cmp_rank"] = pd.to_numeric(out["cmp_rank"], errors="coerce").astype("Int64")
        out["cmp_code"] = [
            cmp_rank_map.get((party, int(rank))) if pd.notna(rank) else pd.NA
            for party, rank in zip(out["party_norm"], out["cmp_rank"])
        ]
        out["cmp_code"] = pd.to_numeric(out["cmp_code"], errors="coerce").astype("Int64")
    else:
        raise ValueError(
            "Claim index needs cmp_code, target_cmp_code, predicted_cmp_code, "
            "or cmp_rank plus a readable --cmp_manifest."
        )

    out = out[out["date_count"].notna() & out["cmp_code"].notna()].copy()
    out["date_count"] = out["date_count"].astype(int)
    out["cmp_code"] = out["cmp_code"].astype(int)
    return out


def compute_manifesto_centroids(manifesto_df: pd.DataFrame, manifesto_emb: np.ndarray) -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    rows = []
    centroids = {}

    for (party, cmp_code), group in manifesto_df.dropna(subset=["cmp_code"]).groupby(["party_norm", "cmp_code"]):
        ids = group["embedding_id"].to_numpy()
        vec = normalize_vec(manifesto_emb[ids].mean(axis=0))
        key = (party, int(cmp_code))
        centroids[key] = vec
        rows.append({
            "party": party,
            "cmp_code": int(cmp_code),
            "n_manifesto_sentences": len(group),
        })

    return pd.DataFrame(rows), centroids


def bootstrap_centroid_ci(
    vectors: np.ndarray,
    own_centroid: np.ndarray,
    other_centroids: list[np.ndarray],
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float, float, float, bool, bool, bool]:
    if len(vectors) == 0 or iterations <= 0:
        return np.nan, np.nan, np.nan, np.nan, False, False, False

    own_scores = np.empty(iterations)
    rel_scores = np.empty(iterations)
    for i in range(iterations):
        sample = vectors[rng.choice(len(vectors), size=len(vectors), replace=True)]
        sample_centroid = normalize_vec(sample.mean(axis=0))
        own_scores[i] = float(sample_centroid @ own_centroid)
        if other_centroids:
            rel_scores[i] = own_scores[i] - float(np.mean([sample_centroid @ c for c in other_centroids]))
        else:
            rel_scores[i] = np.nan

    point_centroid = normalize_vec(vectors.mean(axis=0))
    point_own = float(point_centroid @ own_centroid)
    point_rel = point_own - float(np.mean([point_centroid @ c for c in other_centroids])) if other_centroids else np.nan

    own_low, own_high = np.quantile(own_scores[np.isfinite(own_scores)], [0.025, 0.975])
    if np.isfinite(rel_scores).any():
        rel_low, rel_high = np.quantile(rel_scores[np.isfinite(rel_scores)], [0.025, 0.975])
    else:
        rel_low, rel_high = np.nan, np.nan

    own_contains = bool(own_low <= point_own <= own_high)
    rel_contains = bool(np.isfinite(point_rel) and rel_low <= point_rel <= rel_high) if np.isfinite(rel_low) else False
    return float(own_low), float(own_high), float(rel_low), float(rel_high), True, own_contains, rel_contains


def summarize_counts(claims: pd.DataFrame, output_dir: Path, min_claims: int, dominance_threshold: float) -> None:
    counts = (
        claims.groupby(["party_norm", "cmp_code", "date_count"])
        .size()
        .reset_index(name="n_claims")
        .rename(columns={"party_norm": "party"})
    )
    counts["flag_few_claims"] = counts["n_claims"] < min_claims
    counts.to_csv(output_dir / "claims_per_party_cmp_date_count.csv", index=False)

    stats = (
        counts.groupby(["party", "cmp_code"])["n_claims"]
        .agg(["count", "sum", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_date_counts", "sum": "total_claims"})
    )
    stats["n_few_claim_dates"] = (
        counts[counts["flag_few_claims"]]
        .groupby(["party", "cmp_code"])
        .size()
        .reindex(pd.MultiIndex.from_frame(stats[["party", "cmp_code"]]), fill_value=0)
        .to_numpy()
    )
    stats.to_csv(output_dir / "claim_count_stats_summary.csv", index=False)

    dominance = counts.merge(
        counts.groupby(["party", "cmp_code"])["n_claims"].sum().reset_index(name="total_claims"),
        on=["party", "cmp_code"],
        how="left",
    )
    dominance["share_of_party_cmp_claims"] = dominance["n_claims"] / dominance["total_claims"]
    dominance["flag_dominant_date"] = dominance["share_of_party_cmp_claims"] >= dominance_threshold
    dominance.sort_values(["flag_dominant_date", "share_of_party_cmp_claims"], ascending=[False, False]).to_csv(
        output_dir / "date_count_dominance_check.csv",
        index=False,
    )


def rbf_mmd2_unbiased(x: np.ndarray, y: np.ndarray, sample_size: int, rng: np.random.Generator) -> tuple[float, int]:
    if len(x) < 2 or len(y) < 2:
        return np.nan, min(len(x), len(y))

    sample_size_used = min(sample_size, len(x), len(y))
    if len(x) > sample_size_used:
        x = x[rng.choice(len(x), size=sample_size_used, replace=False)]
    if len(y) > sample_size_used:
        y = y[rng.choice(len(y), size=sample_size_used, replace=False)]

    combined = np.vstack([x, y])
    if len(combined) > 1000:
        combined = combined[rng.choice(len(combined), size=1000, replace=False)]

    dots = combined @ combined.T
    sq = np.clip(2 - 2 * dots, 0, None)
    median_sq = np.median(sq[sq > 0]) if np.any(sq > 0) else 1.0
    gamma = 1.0 / (2.0 * median_sq)

    def kernel_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dist_sq = np.clip(2 - 2 * (a @ b.T), 0, None)
        return np.exp(-gamma * dist_sq)

    kxx = kernel_matrix(x, x)
    kyy = kernel_matrix(y, y)
    kxy = kernel_matrix(x, y)
    n = len(x)
    m = len(y)
    kxx_sum = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    kyy_sum = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    return float(kxx_sum + kyy_sum - 2 * kxy.mean()), sample_size_used


def aggregate_centroid_alignment(
    claims: pd.DataFrame,
    claim_emb: np.ndarray,
    manifesto_df: pd.DataFrame,
    manifesto_emb: np.ndarray,
    manifesto_centroids: dict[tuple[str, int], np.ndarray],
    output_dir: Path,
    bootstrap_iters: int,
    seed: int,
    mmd_sample_size: int,
    min_claims_per_date: int,
    min_bootstrap_claims: int,
) -> tuple[pd.DataFrame, dict[tuple[str, int, int], np.ndarray]]:
    rng = np.random.default_rng(seed)
    manifesto_lookup = {
        (party, int(cmp_code)): group["embedding_id"].to_numpy()
        for (party, cmp_code), group in manifesto_df.dropna(subset=["cmp_code"]).groupby(["party_norm", "cmp_code"])
    }

    rows = []
    date_centroids = {}
    for keys, group in claims.groupby(["party_norm", "cmp_code", "date_count"]):
        party, cmp_code, date_count = keys
        own_centroid = manifesto_centroids.get((party, int(cmp_code)))
        if own_centroid is None:
            continue

        claim_ids = group["embedding_id"].to_numpy(dtype=int)
        claim_vectors = claim_emb[claim_ids]
        debate_centroid = normalize_vec(claim_vectors.mean(axis=0))
        date_centroids[(party, int(cmp_code), int(date_count))] = debate_centroid

        other_centroids = [
            centroid
            for (other_party, other_cmp), centroid in manifesto_centroids.items()
            if other_cmp == int(cmp_code) and other_party != party
        ]

        own_similarity = float(debate_centroid @ own_centroid)
        if other_centroids:
            avg_other_similarity = float(np.mean([debate_centroid @ centroid for centroid in other_centroids]))
            relative_alignment = own_similarity - avg_other_similarity
        else:
            avg_other_similarity = np.nan
            relative_alignment = np.nan

        if len(group) >= min_bootstrap_claims:
            (
                own_ci_low,
                own_ci_high,
                rel_ci_low,
                rel_ci_high,
                bootstrap_valid,
                own_ci_contains_point,
                relative_ci_contains_point,
            ) = bootstrap_centroid_ci(
                vectors=claim_vectors,
                own_centroid=own_centroid,
                other_centroids=other_centroids,
                rng=rng,
                iterations=bootstrap_iters,
            )
        else:
            own_ci_low = own_ci_high = rel_ci_low = rel_ci_high = np.nan
            bootstrap_valid = False
            own_ci_contains_point = False
            relative_ci_contains_point = False

        mani_ids = manifesto_lookup.get((party, int(cmp_code)))
        if mani_ids is None:
            mmd = np.nan
            mmd_sample_size_used = 0
            n_manifesto_sentences = 0
        else:
            n_manifesto_sentences = len(mani_ids)
            mmd, mmd_sample_size_used = rbf_mmd2_unbiased(claim_emb[claim_ids], manifesto_emb[mani_ids], mmd_sample_size, rng)

        rows.append({
            "party": party,
            "cmp_code": int(cmp_code),
            "date_count": int(date_count),
            "n_claims": len(group),
            "flag_few_claims": len(group) < min_claims_per_date,
            "centroid_own_manifesto_similarity": own_similarity,
            "own_similarity_ci_low": own_ci_low,
            "own_similarity_ci_high": own_ci_high,
            "centroid_relative_alignment": relative_alignment,
            "relative_alignment_ci_low": rel_ci_low,
            "relative_alignment_ci_high": rel_ci_high,
            "mmd_to_own_manifesto": mmd,
            "mmd_sample_size_used": mmd_sample_size_used,
            "n_manifesto_sentences": n_manifesto_sentences,
            "centroid_avg_other_manifesto_similarity": avg_other_similarity,
            "n_other_manifesto_centroids": len(other_centroids),
            "bootstrap_valid": bootstrap_valid,
            "own_ci_contains_point": own_ci_contains_point,
            "relative_ci_contains_point": relative_ci_contains_point,
        })

    out = pd.DataFrame(rows).sort_values(["party", "cmp_code", "date_count"])
    out.to_csv(output_dir / "alignment_by_party_cmp_date_count.csv", index=False)
    return out, date_centroids


def trend_summary(aggregate: pd.DataFrame, output_dir: Path, min_dates: int = 3) -> pd.DataFrame:
    rows = []
    for (party, cmp_code), group in aggregate.groupby(["party", "cmp_code"]):
        group = group.sort_values("date_count")
        x = group["date_count"].to_numpy(dtype=float)

        for metric in ["centroid_own_manifesto_similarity", "centroid_relative_alignment", "mmd_to_own_manifesto"]:
            y = group[metric].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < min_dates or len(np.unique(x[valid])) < 2:
                slope = intercept = spearman = np.nan
            else:
                slope, intercept = np.polyfit(x[valid], y[valid], 1)
                spearman = pd.Series(x[valid]).corr(pd.Series(y[valid]), method="spearman")

            rows.append({
                "party": party,
                "cmp_code": cmp_code,
                "metric": metric,
                "n_date_counts": int(valid.sum()),
                "linear_slope": slope,
                "linear_intercept": intercept,
                "spearman_correlation": spearman,
            })

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "trend_summary_by_party_cmp.csv", index=False)
    return out


def bootstrap_diagnostics(aggregate: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for metric, flag_col in [
        ("centroid_own_manifesto_similarity", "own_ci_contains_point"),
        ("centroid_relative_alignment", "relative_ci_contains_point"),
    ]:
        valid = aggregate[aggregate["bootstrap_valid"]].copy()
        rows.append({
            "metric": metric,
            "n_bootstrap_valid_cells": int(len(valid)),
            "n_point_outside_ci": int((~valid[flag_col]).sum()) if not valid.empty else 0,
            "share_point_outside_ci": float((~valid[flag_col]).mean()) if not valid.empty else np.nan,
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "bootstrap_ci_diagnostics.csv", index=False)
    return out


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def weighted_std(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    avg = np.average(values[valid], weights=weights[valid])
    var = np.average((values[valid] - avg) ** 2, weights=weights[valid])
    return float(np.sqrt(var))


def harmonic_mean(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return float(2 * a * b / (a + b))


def compute_manifesto_anchored_drift(
    aggregate: pd.DataFrame,
    date_centroids: dict[tuple[str, int, int], np.ndarray],
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for (party, cmp_code), group in aggregate.groupby(["party", "cmp_code"]):
        group = group.sort_values("date_count")
        previous_row = None
        previous_vec = None

        for _, row in group.iterrows():
            key = (party, int(cmp_code), int(row["date_count"]))
            vec = date_centroids.get(key)
            if previous_row is not None and previous_vec is not None and vec is not None:
                n_prev = int(previous_row["n_claims"])
                n_current = int(row["n_claims"])
                transition_weight = harmonic_mean(n_prev, n_current)
                previous_alignment = float(previous_row["centroid_own_manifesto_similarity"])
                current_alignment = float(row["centroid_own_manifesto_similarity"])
                signed_alignment_change = current_alignment - previous_alignment
                manifesto_drift = abs(signed_alignment_change)
                local_drift = 1 - float(vec @ previous_vec)
                rows.append({
                    "party": party,
                    "cmp_code": int(cmp_code),
                    "previous_date_count": int(previous_row["date_count"]),
                    "date_count": int(row["date_count"]),
                    "date_gap": int(row["date_count"] - previous_row["date_count"]),
                    "previous_manifesto_alignment": previous_alignment,
                    "current_manifesto_alignment": current_alignment,
                    "signed_manifesto_alignment_change": signed_alignment_change,
                    "manifesto_anchored_drift": manifesto_drift,
                    "local_debate_drift": local_drift,
                    "n_claims_prev": n_prev,
                    "n_claims_current": n_current,
                    "transition_weight": transition_weight,
                })

            previous_row = row
            previous_vec = vec

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "manifesto_anchored_drift.csv", index=False)
    return out


def party_date_alignment(drift: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    if drift.empty:
        out = pd.DataFrame(columns=[
            "party",
            "date_count",
            "weighted_manifesto_anchored_drift",
            "weighted_local_drift",
            "n_cmp_codes",
            "total_transition_weight",
            "mean_n_claims_current",
            "mean_n_claims_prev",
            "mean_date_gap",
        ])
        out.to_csv(output_dir / "alignment_by_party_date_count.csv", index=False)
        return out

    for (party, date_count), group in drift.groupby(["party", "date_count"]):
        weights = group["transition_weight"]
        rows.append({
            "party": party,
            "date_count": int(date_count),
            "weighted_manifesto_anchored_drift": weighted_average(group["manifesto_anchored_drift"], weights),
            "weighted_local_drift": weighted_average(group["local_debate_drift"], weights),
            "n_cmp_codes": int(group["cmp_code"].nunique()),
            "total_transition_weight": float(weights.sum()),
            "mean_n_claims_current": weighted_average(group["n_claims_current"], weights),
            "mean_n_claims_prev": weighted_average(group["n_claims_prev"], weights),
            "mean_date_gap": weighted_average(group["date_gap"], weights),
        })

    out = pd.DataFrame(rows).sort_values(["party", "date_count"])
    out.to_csv(output_dir / "alignment_by_party_date_count.csv", index=False)
    return out


def party_cycle_drift_summary(drift: pd.DataFrame, aggregate: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    if drift.empty:
        out = pd.DataFrame(columns=[
            "party",
            "weighted_mean_manifesto_drift",
            "weighted_mean_local_drift",
            "weighted_std_manifesto_drift",
            "weighted_std_local_drift",
            "total_transitions",
            "total_claims",
        ])
        out.to_csv(output_dir / "party_cycle_drift_summary.csv", index=False)
        return out

    total_claims_by_party = aggregate.groupby("party")["n_claims"].sum()
    for party, group in drift.groupby("party"):
        weights = group["transition_weight"]
        rows.append({
            "party": party,
            "weighted_mean_manifesto_drift": weighted_average(group["manifesto_anchored_drift"], weights),
            "weighted_mean_local_drift": weighted_average(group["local_debate_drift"], weights),
            "weighted_std_manifesto_drift": weighted_std(group["manifesto_anchored_drift"], weights),
            "weighted_std_local_drift": weighted_std(group["local_debate_drift"], weights),
            "total_transitions": int(len(group)),
            "total_claims": int(total_claims_by_party.get(party, 0)),
        })

    out = pd.DataFrame(rows).sort_values("party")
    out.to_csv(output_dir / "party_cycle_drift_summary.csv", index=False)
    return out


def add_period_labels(aggregate: pd.DataFrame, early_q: float, election_q: float) -> pd.DataFrame:
    out = aggregate.copy()
    out["period"] = "mid_term"
    for party, idx in out.groupby("party").groups.items():
        party_dates = out.loc[idx, "date_count"]
        early_cut = party_dates.quantile(early_q)
        election_cut = party_dates.quantile(election_q)
        out.loc[party_dates[party_dates <= early_cut].index, "period"] = "early_term"
        out.loc[party_dates[party_dates >= election_cut].index, "period"] = "election_period"
    return out


def period_comparisons(aggregate: pd.DataFrame, output_dir: Path, early_q: float, election_q: float) -> pd.DataFrame:
    with_period = add_period_labels(aggregate, early_q, election_q)
    with_period.to_csv(output_dir / "alignment_by_party_cmp_date_count_with_period.csv", index=False)

    metrics = [
        "centroid_own_manifesto_similarity",
        "centroid_relative_alignment",
        "mmd_to_own_manifesto",
    ]

    rows_cmp = []
    for (party, cmp_code, period), group in with_period.groupby(["party", "cmp_code", "period"]):
        weights = group["n_claims"]
        row = {
            "party": party,
            "cmp_code": int(cmp_code),
            "period": period,
            "total_claims": int(weights.sum()),
            "n_date_counts": int(group["date_count"].nunique()),
        }
        for metric in metrics:
            row[f"weighted_{metric}"] = weighted_average(group[metric], weights)
            row[f"equal_weight_{metric}"] = float(group[metric].mean())
        rows_cmp.append(row)

    out_cmp = pd.DataFrame(rows_cmp).sort_values(["party", "cmp_code", "period"])
    out_cmp.to_csv(output_dir / "period_comparison_by_party_cmp.csv", index=False)

    rows_party = []
    for (party, period), group in with_period.groupby(["party", "period"]):
        weights = group["n_claims"]
        row = {
            "party": party,
            "period": period,
            "total_claims": int(weights.sum()),
            "n_cmp_codes": int(group["cmp_code"].nunique()),
            "n_date_counts": int(group["date_count"].nunique()),
        }
        for metric in metrics:
            row[f"weighted_{metric}"] = weighted_average(group[metric], weights)
            row[f"equal_weight_{metric}"] = float(group[metric].mean())
        rows_party.append(row)

    out_party = pd.DataFrame(rows_party).sort_values(["party", "period"])
    out_party.to_csv(output_dir / "period_comparison_by_party.csv", index=False)
    return with_period


def debate_to_debate_drift(aggregate: pd.DataFrame, date_centroids: dict[tuple[str, int, int], np.ndarray], output_dir: Path) -> pd.DataFrame:
    rows = []
    for (party, cmp_code), group in aggregate.groupby(["party", "cmp_code"]):
        group = group.sort_values("date_count")
        previous_row = None
        previous_vec = None
        for _, row in group.iterrows():
            key = (party, int(cmp_code), int(row["date_count"]))
            vec = date_centroids.get(key)
            if previous_row is not None and previous_vec is not None and vec is not None:
                cosine_similarity = float(vec @ previous_vec)
                rows.append({
                    "party": party,
                    "cmp_code": int(cmp_code),
                    "date_count": int(row["date_count"]),
                    "previous_date_count": int(previous_row["date_count"]),
                    "n_claims": int(row["n_claims"]),
                    "previous_n_claims": int(previous_row["n_claims"]),
                    "debate_to_debate_cosine_distance": 1 - cosine_similarity,
                })
            previous_row = row
            previous_vec = vec

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "debate_to_debate_drift.csv", index=False)

    party_rows = []
    if not out.empty:
        for (party, date_count), group in out.groupby(["party", "date_count"]):
            weights = group["n_claims"]
            party_rows.append({
                "party": party,
                "date_count": int(date_count),
                "total_claims": int(weights.sum()),
                "n_cmp_codes": int(group["cmp_code"].nunique()),
                "weighted_debate_to_debate_cosine_distance": weighted_average(group["debate_to_debate_cosine_distance"], weights),
                "equal_weight_debate_to_debate_cosine_distance": float(group["debate_to_debate_cosine_distance"].mean()),
            })

    party_out = pd.DataFrame(party_rows)
    party_out.to_csv(output_dir / "debate_to_debate_drift_by_party_date_count.csv", index=False)
    return out


def adaptive_weighted_smooth(x: np.ndarray, y: np.ndarray, weights: np.ndarray, min_claims: int) -> np.ndarray:
    smooth = np.full(len(y), np.nan)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 3:
        return smooth

    xv = x[valid]
    yv = y[valid]
    wv = weights[valid]
    valid_positions = np.where(valid)[0]

    for target_pos, target_x in zip(valid_positions, xv):
        order = np.argsort(np.abs(xv - target_x))
        cumulative = np.cumsum(wv[order])
        take_n = int(np.searchsorted(cumulative, min_claims, side="left") + 1)
        take_n = max(3, min(take_n, len(order)))
        chosen = order[:take_n]
        xs = xv[chosen]
        ys = yv[chosen]
        ws = wv[chosen]

        max_dist = np.max(np.abs(xs - target_x))
        if max_dist > 0:
            local = (1 - (np.abs(xs - target_x) / max_dist) ** 3) ** 3
            ws = ws * local

        if ws.sum() <= 0:
            continue

        design = np.column_stack([np.ones(len(xs)), xs - target_x])
        try:
            sqrt_w = np.sqrt(ws)
            beta = np.linalg.lstsq(design * sqrt_w[:, None], ys * sqrt_w, rcond=None)[0]
            smooth[target_pos] = beta[0]
        except np.linalg.LinAlgError:
            smooth[target_pos] = np.average(ys, weights=ws)

    return smooth


def weighted_lowess(x: np.ndarray, y: np.ndarray, weights: np.ndarray, frac: float) -> np.ndarray:
    smooth = np.full(len(y), np.nan)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 3:
        return smooth

    xv = x[valid]
    yv = y[valid]
    wv = weights[valid]
    valid_positions = np.where(valid)[0]
    window = max(3, int(np.ceil(frac * len(xv))))

    for target_pos, target_x in zip(valid_positions, xv):
        order = np.argsort(np.abs(xv - target_x))[:window]
        xs = xv[order]
        ys = yv[order]
        ws = wv[order].copy()
        max_dist = np.max(np.abs(xs - target_x))
        if max_dist > 0:
            ws *= (1 - (np.abs(xs - target_x) / max_dist) ** 3) ** 3
        if ws.sum() <= 0:
            continue
        design = np.column_stack([np.ones(len(xs)), xs - target_x])
        try:
            sqrt_w = np.sqrt(ws)
            beta = np.linalg.lstsq(design * sqrt_w[:, None], ys * sqrt_w, rcond=None)[0]
            smooth[target_pos] = beta[0]
        except np.linalg.LinAlgError:
            smooth[target_pos] = np.average(ys, weights=ws)

    return smooth


def binned_party_drift(party_df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    if party_df.empty:
        return pd.DataFrame()
    date_min = party_df["date_count"].min()
    date_max = party_df["date_count"].max()
    if date_min == date_max:
        return pd.DataFrame([{
            "bin_midpoint": float(date_min),
            "weighted_manifesto_anchored_drift": weighted_average(
                party_df["weighted_manifesto_anchored_drift"],
                party_df["total_transition_weight"],
            ),
            "weighted_local_drift": weighted_average(
                party_df["weighted_local_drift"],
                party_df["total_transition_weight"],
            ),
            "total_transition_weight": float(party_df["total_transition_weight"].sum()),
            "mean_n_claims_current": weighted_average(party_df["mean_n_claims_current"], party_df["total_transition_weight"]),
            "mean_n_claims_prev": weighted_average(party_df["mean_n_claims_prev"], party_df["total_transition_weight"]),
        }])

    bins = np.linspace(date_min, date_max, n_bins + 1)
    out = party_df.copy()
    out["bin"] = pd.cut(out["date_count"], bins=bins, include_lowest=True, labels=False)
    rows = []
    for bin_id, group in out.groupby("bin", dropna=True):
        if group.empty:
            continue
        weights = group["total_transition_weight"]
        left = bins[int(bin_id)]
        right = bins[int(bin_id) + 1]
        rows.append({
            "bin": int(bin_id),
            "bin_midpoint": float((left + right) / 2),
            "weighted_manifesto_anchored_drift": weighted_average(group["weighted_manifesto_anchored_drift"], weights),
            "weighted_local_drift": weighted_average(group["weighted_local_drift"], weights),
            "total_transition_weight": float(weights.sum()),
            "mean_n_claims_current": weighted_average(group["mean_n_claims_current"], weights),
            "mean_n_claims_prev": weighted_average(group["mean_n_claims_prev"], weights),
        })
    return pd.DataFrame(rows).sort_values("bin_midpoint")


def event_markers_from_debates(debates_csv: Path, output_dir: Path, cabinet_start_date: str, pre_election_date: str) -> pd.DataFrame:
    labels = [
        ("cabinet_start", cabinet_start_date),
        ("six_months_before_election", pre_election_date),
    ]
    columns = ["event", "event_date", "date_count", "matched_meeting_date"]

    if not debates_csv.exists():
        out = pd.DataFrame(columns=columns)
        out.to_csv(output_dir / "event_date_count_markers.csv", index=False)
        return out

    debates = pd.read_csv(debates_csv)
    if "foi_meetingDate" not in debates.columns or "day_count" not in debates.columns:
        out = pd.DataFrame(columns=columns)
        out.to_csv(output_dir / "event_date_count_markers.csv", index=False)
        return out

    dates = debates[["foi_meetingDate", "day_count"]].dropna().copy()
    dates["foi_meetingDate"] = pd.to_datetime(dates["foi_meetingDate"], errors="coerce")
    dates["day_count"] = pd.to_numeric(dates["day_count"], errors="coerce")
    dates = dates.dropna().drop_duplicates("foi_meetingDate").sort_values("foi_meetingDate")
    dates = dates[dates["day_count"] >= 0].copy()

    rows = []
    for event, date_text in labels:
        event_date = pd.to_datetime(date_text)
        if dates.empty:
            rows.append({
                "event": event,
                "event_date": event_date.date().isoformat(),
                "date_count": np.nan,
                "matched_meeting_date": "",
            })
            continue

        nearest_idx = (dates["foi_meetingDate"] - event_date).abs().idxmin()
        nearest = dates.loc[nearest_idx]
        rows.append({
            "event": event,
            "event_date": event_date.date().isoformat(),
            "date_count": int(nearest["day_count"]),
            "matched_meeting_date": nearest["foi_meetingDate"].date().isoformat(),
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "event_date_count_markers.csv", index=False)
    return out


def plot_party_cycle_drift(
    party_date: pd.DataFrame,
    output_dir: Path,
    min_claims_for_smoothing: int,
    lowess_frac: float,
    n_temporal_bins: int,
    event_markers: pd.DataFrame,
) -> None:
    plot_dir = output_dir / "plots" / "manifesto_anchored_drift"
    plot_dir.mkdir(parents=True, exist_ok=True)

    binned_rows = []
    for party, group in party_date.groupby("party"):
        group = group.sort_values("date_count")
        if group.empty:
            continue

        binned = binned_party_drift(group, n_temporal_bins)
        if not binned.empty:
            binned["party"] = party
            binned_rows.append(binned)

        plt.figure(figsize=(10, 5))

        for _, marker in event_markers.dropna(subset=["date_count"]).iterrows():
            label = (
                "cabinet start"
                if marker["event"] == "cabinet_start"
                else "six months before election"
            )
            plt.axvline(
                marker["date_count"],
                color="0.35",
                linewidth=1.0,
                alpha=0.45,
                linestyle="--",
                label=label,
            )

        if not binned.empty:
            bx = binned["bin_midpoint"].to_numpy()
            by = binned["weighted_manifesto_anchored_drift"].to_numpy(dtype=float)
            bw = binned["total_transition_weight"].to_numpy(dtype=float)
            stable = binned["total_transition_weight"].to_numpy() >= min_claims_for_smoothing
            if np.nanmax(bw) > np.nanmin(bw):
                sizes = 55 + 130 * (bw - np.nanmin(bw)) / (np.nanmax(bw) - np.nanmin(bw))
            else:
                sizes = np.full(len(bw), 80.0)
            plt.scatter(
                bx,
                by,
                s=sizes,
                marker="o",
                alpha=0.82,
                color="#2f6f9f",
                edgecolor="white",
                linewidth=0.8,
                label="weighted binned means",
            )
            smooth = weighted_lowess(bx[stable], by[stable], bw[stable], lowess_frac)
            if np.isfinite(smooth).sum() >= 2:
                order = np.argsort(bx[stable])
                plt.plot(
                    bx[stable][order],
                    smooth[order],
                    color="#111111",
                    linewidth=3.0,
                    label=f"weighted LOWESS trend (frac={lowess_frac})",
                )

        plt.title(f"{party}: manifesto-anchored argument drift over the political cycle")
        plt.xlabel("date_count")
        plt.ylabel("weighted manifesto-anchored drift")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{party}_manifesto_anchored_drift.png", dpi=200)
        plt.close()

    if binned_rows:
        pd.concat(binned_rows, ignore_index=True).to_csv(output_dir / "party_drift_binned_for_plotting.csv", index=False)
    else:
        pd.DataFrame(columns=[
            "bin",
            "bin_midpoint",
            "weighted_manifesto_anchored_drift",
            "weighted_local_drift",
            "total_transition_weight",
            "mean_n_claims_current",
            "mean_n_claims_prev",
            "party",
        ]).to_csv(output_dir / "party_drift_binned_for_plotting.csv", index=False)


def plot_metric_series(aggregate: pd.DataFrame, output_dir: Path, metric: str, ci_cols: tuple[str, str] | None, max_series: int, min_claims_for_smoothing: int) -> None:
    plot_dir = output_dir / "plots" / metric
    plot_dir.mkdir(parents=True, exist_ok=True)

    series_sizes = (
        aggregate.groupby(["party", "cmp_code"])["n_claims"]
        .sum()
        .sort_values(ascending=False)
        .head(max_series)
    )

    for party, cmp_code in series_sizes.index:
        group = aggregate[(aggregate["party"] == party) & (aggregate["cmp_code"] == cmp_code)].sort_values("date_count")
        if len(group) < 2:
            continue

        x = group["date_count"].to_numpy()
        y = group[metric].to_numpy(dtype=float)
        weights = group["n_claims"].to_numpy(dtype=float)
        low_claims = group["flag_few_claims"].to_numpy(dtype=bool)

        plt.figure(figsize=(10, 5))
        plt.scatter(x[~low_claims], y[~low_claims], s=34, label="raw points")
        if low_claims.any():
            plt.scatter(x[low_claims], y[low_claims], s=42, marker="x", label="low-claim points")

        if ci_cols is not None:
            low = group[ci_cols[0]].to_numpy(dtype=float)
            high = group[ci_cols[1]].to_numpy(dtype=float)
            ci_valid = group["bootstrap_valid"].to_numpy(dtype=bool)
            if ci_valid.any():
                plt.fill_between(x[ci_valid], low[ci_valid], high[ci_valid], alpha=0.2, label="95% bootstrap CI")

        smooth = adaptive_weighted_smooth(x, y, weights, min_claims_for_smoothing)
        if np.isfinite(smooth).sum() >= 2:
            plt.plot(x, smooth, linewidth=2.0, label=f"adaptive weighted smooth ({min_claims_for_smoothing}+ claims)")

        plt.title(f"{party} CMP {cmp_code}: {metric} over date_count")
        plt.xlabel("date_count")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{party}_cmp_{cmp_code}_{metric}.png", dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plots").mkdir(parents=True, exist_ok=True)

    prefilter_root = resolve_prefilter_root(args.prefilter_root)
    manifesto_index, manifesto_embeddings = resolve_manifesto_paths(args.manifesto_index, args.manifesto_embeddings)
    parties = parse_parties(args.parties)

    claims_raw, claim_emb = read_party_embeddings(prefilter_root, parties)
    cmp_rank_map = load_cmp_rank_map(args.cmp_manifest)
    claims = prepare_claims(claims_raw, cmp_rank_map)
    manifesto_df, manifesto_emb = load_manifesto(manifesto_index, manifesto_embeddings)

    _, centroids = compute_manifesto_centroids(manifesto_df, manifesto_emb)
    summarize_counts(claims, args.output_dir, args.min_claims_per_date, args.dominance_threshold)

    stale_claim_level = args.output_dir / "claim_level_manifesto_alignment.csv"
    if stale_claim_level.exists():
        stale_claim_level.unlink()

    aggregate, date_centroids = aggregate_centroid_alignment(
        claims=claims,
        claim_emb=claim_emb,
        manifesto_df=manifesto_df,
        manifesto_emb=manifesto_emb,
        manifesto_centroids=centroids,
        output_dir=args.output_dir,
        bootstrap_iters=args.bootstrap_iters,
        seed=args.bootstrap_seed,
        mmd_sample_size=args.mmd_sample_size,
        min_claims_per_date=args.min_claims_per_date,
        min_bootstrap_claims=args.min_bootstrap_claims,
    )
    if aggregate.empty:
        raise ValueError("No party-CMP-date centroids could be scored against same-party manifesto CMP centroids.")

    event_markers = event_markers_from_debates(
        debates_csv=args.debates_csv,
        output_dir=args.output_dir,
        cabinet_start_date=args.cabinet_start_date,
        pre_election_date=args.pre_election_date,
    )
    drift = compute_manifesto_anchored_drift(aggregate, date_centroids, args.output_dir)
    party_date = party_date_alignment(drift, args.output_dir)
    party_cycle_drift_summary(drift, aggregate, args.output_dir)
    bootstrap_diagnostics(aggregate, args.output_dir)

    plot_party_cycle_drift(
        party_date=party_date,
        output_dir=args.output_dir,
        min_claims_for_smoothing=args.min_claims_for_smoothing,
        lowess_frac=args.lowess_frac,
        n_temporal_bins=args.n_temporal_bins,
        event_markers=event_markers,
    )

    print(f"Saved analysis outputs to: {args.output_dir}")
    print(f"Party-CMP-date rows: {len(aggregate)}")
    print(f"Manifesto-anchored drift transitions: {len(drift)}")


if __name__ == "__main__":
    main()
