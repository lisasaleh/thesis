import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFILTER_ROOT = Path("/scratch_shared/lsaleh/prefilter_pipeline")
ALT_PREFILTER_ROOT = Path("/scratch-shared/lsaleh/prefilter_pipeline")
DEFAULT_MANIFESTO_INDEX = ROOT / "outputs/embeddings/manifesto_sbert_embedding_index.csv"
DEFAULT_MANIFESTO_EMB = ROOT / "outputs/embeddings/manifesto_sbert_embeddings.npy"
ALT_MANIFESTO_INDEX = ROOT / "outputs/manifesto/embeddings/manifesto_sbert_embedding_index.csv"
ALT_MANIFESTO_EMB = ROOT / "outputs/manifesto/embeddings/manifesto_sbert_embeddings.npy"
DEFAULT_CMP_MANIFEST = ROOT / "outputs/cmp_manifest.csv"
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
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min_claims_per_date", type=int, default=5)
    parser.add_argument("--dominance_threshold", type=float, default=0.25)
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=13)
    parser.add_argument("--mmd_sample_size", type=int, default=500)
    parser.add_argument("--rolling_window", type=int, default=5)
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
) -> tuple[float, float, float, float]:
    if len(vectors) == 0:
        return np.nan, np.nan, np.nan, np.nan
    if len(vectors) == 1 or iterations <= 0:
        centroid = normalize_vec(vectors.mean(axis=0))
        own = float(centroid @ own_centroid)
        rel = own - float(np.mean([centroid @ c for c in other_centroids])) if other_centroids else np.nan
        return own, own, rel, rel

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

    own_low, own_high = np.quantile(own_scores[np.isfinite(own_scores)], [0.025, 0.975])
    if np.isfinite(rel_scores).any():
        rel_low, rel_high = np.quantile(rel_scores[np.isfinite(rel_scores)], [0.025, 0.975])
    else:
        rel_low, rel_high = np.nan, np.nan
    return float(own_low), float(own_high), float(rel_low), float(rel_high)


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


def rbf_mmd2(x: np.ndarray, y: np.ndarray, sample_size: int, rng: np.random.Generator) -> float:
    if len(x) == 0 or len(y) == 0:
        return np.nan
    if len(x) > sample_size:
        x = x[rng.choice(len(x), size=sample_size, replace=False)]
    if len(y) > sample_size:
        y = y[rng.choice(len(y), size=sample_size, replace=False)]

    combined = np.vstack([x, y])
    if len(combined) > 1000:
        combined = combined[rng.choice(len(combined), size=1000, replace=False)]

    dots = combined @ combined.T
    sq = np.clip(2 - 2 * dots, 0, None)
    median_sq = np.median(sq[sq > 0]) if np.any(sq > 0) else 1.0
    gamma = 1.0 / (2.0 * median_sq)

    def kernel_mean(a: np.ndarray, b: np.ndarray) -> float:
        dist_sq = np.clip(2 - 2 * (a @ b.T), 0, None)
        return float(np.exp(-gamma * dist_sq).mean())

    return kernel_mean(x, x) + kernel_mean(y, y) - 2 * kernel_mean(x, y)


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
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    manifesto_lookup = {
        (party, int(cmp_code)): group["embedding_id"].to_numpy()
        for (party, cmp_code), group in manifesto_df.dropna(subset=["cmp_code"]).groupby(["party_norm", "cmp_code"])
    }

    rows = []
    for keys, group in claims.groupby(["party_norm", "cmp_code", "date_count"]):
        party, cmp_code, date_count = keys
        own_centroid = manifesto_centroids.get((party, int(cmp_code)))
        if own_centroid is None:
            continue

        claim_ids = group["embedding_id"].to_numpy(dtype=int)
        claim_vectors = claim_emb[claim_ids]
        debate_centroid = normalize_vec(claim_vectors.mean(axis=0))

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

        own_ci_low, own_ci_high, rel_ci_low, rel_ci_high = bootstrap_centroid_ci(
            vectors=claim_vectors,
            own_centroid=own_centroid,
            other_centroids=other_centroids,
            rng=rng,
            iterations=bootstrap_iters,
        )

        mani_ids = manifesto_lookup.get((party, int(cmp_code)))
        if mani_ids is None:
            mmd = np.nan
        else:
            mmd = rbf_mmd2(claim_emb[claim_ids], manifesto_emb[mani_ids], mmd_sample_size, rng)

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
            "centroid_avg_other_manifesto_similarity": avg_other_similarity,
            "n_other_manifesto_centroids": len(other_centroids),
        })

    out = pd.DataFrame(rows).sort_values(["party", "cmp_code", "date_count"])
    out.to_csv(output_dir / "alignment_by_party_cmp_date_count.csv", index=False)
    return out


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


def plot_metric_series(aggregate: pd.DataFrame, output_dir: Path, metric: str, ci_cols: tuple[str, str] | None, max_series: int, rolling_window: int) -> None:
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

        plt.figure(figsize=(10, 5))
        plt.plot(x, y, marker="o", linewidth=1.5, label=metric)

        if ci_cols is not None:
            low = group[ci_cols[0]].to_numpy(dtype=float)
            high = group[ci_cols[1]].to_numpy(dtype=float)
            plt.fill_between(x, low, high, alpha=0.2, label="95% bootstrap CI")

        if rolling_window > 1 and len(group) >= rolling_window:
            rolling = group[metric].rolling(rolling_window, min_periods=max(2, rolling_window // 2)).mean()
            plt.plot(x, rolling, linewidth=2.0, label=f"rolling mean ({rolling_window})")

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

    aggregate = aggregate_centroid_alignment(
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
    )
    if aggregate.empty:
        raise ValueError("No party-CMP-date centroids could be scored against same-party manifesto CMP centroids.")

    trend_summary(aggregate, args.output_dir)

    plot_metric_series(
        aggregate,
        args.output_dir,
        "centroid_own_manifesto_similarity",
        ("own_similarity_ci_low", "own_similarity_ci_high"),
        args.max_plot_series,
        args.rolling_window,
    )
    plot_metric_series(
        aggregate,
        args.output_dir,
        "centroid_relative_alignment",
        ("relative_alignment_ci_low", "relative_alignment_ci_high"),
        args.max_plot_series,
        args.rolling_window,
    )
    plot_metric_series(
        aggregate,
        args.output_dir,
        "mmd_to_own_manifesto",
        None,
        args.max_plot_series,
        args.rolling_window,
    )

    print(f"Saved analysis outputs to: {args.output_dir}")
    print(f"Party-CMP-date rows: {len(aggregate)}")


if __name__ == "__main__":
    main()
