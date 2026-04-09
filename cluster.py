import os
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import umap
import hdbscan


class Embedder:
    """Generate embeddings using SentenceTransformers."""
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, batch_size=64, normalize=True):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )


def reduce_embeddings(embeddings, n_neighbors=8, n_components=10, min_dist=0.0, metric="cosine", random_state=42):
    """Reduce embeddings dimensionality using UMAP."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced


def cluster_hdbscan(reduced_embeddings, min_cluster_size=3, min_samples=1):
    """Cluster embeddings using HDBSCAN."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(reduced_embeddings)
    probabilities = clusterer.probabilities_
    return labels, probabilities, clusterer


def load_and_prepare_data(input_csv):
    """Load and clean data for clustering."""
    df = pd.read_csv(input_csv)

    df["point_clean"] = (
        df["point"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df = df[df["point_clean"] != ""].copy()
    df = df[df["point_clean"].str.len() > 10].copy()

    df["point_uid"] = (
        df["document_id"].astype(str) + "_"
        + df["intervention_id"].astype(str) + "_"
        + df["claim_idx"].astype(str)
    )

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster points using embeddings, UMAP, and HDBSCAN.")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV with points to cluster")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                        help="SentenceTransformer model name")
    parser.add_argument("--add_timestamp", action="store_true", help="Add timestamp to output filenames")
    
    # HDBSCAN parameters
    parser.add_argument("--min_cluster_size", type=int, default=3, help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--min_samples", type=int, default=1, help="Minimum samples for HDBSCAN")
    
    # UMAP parameters
    parser.add_argument("--n_neighbors", type=int, default=8, help="Number of neighbors for UMAP")
    parser.add_argument("--min_dist", type=float, default=0.0, help="Minimum distance for UMAP")
    
    return parser.parse_args()


def main(input_csv, output_dir, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 
         add_timestamp=False, min_cluster_size=3, min_samples=1, n_neighbors=8, min_dist=0.0):
    """Main clustering pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with optional timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
    timestamp_str = f"_{timestamp}" if timestamp else ""
    
    clustered_file = os.path.join(output_dir, f"clustered_points{timestamp_str}.csv")

    # Load and prepare data
    df = load_and_prepare_data(input_csv)

    # Cluster on unique points
    df_unique = df.drop_duplicates(subset=["point_clean"]).copy().reset_index(drop=True)

    # Generate embeddings
    print("[DEBUG] Loading embedding model...", flush=True)
    embedder = Embedder(model_name=model_name)
    embeddings = embedder.encode(df_unique["point_clean"].tolist())
    print("[DEBUG] Embeddings generated.", flush=True)

    # Reduce dimensionality
    print("[DEBUG] Reducing dimensionality with UMAP...", flush=True)
    reduced = reduce_embeddings(embeddings, n_neighbors=n_neighbors, min_dist=min_dist)
    print("[DEBUG] Dimensionality reduction complete.", flush=True)

    # Cluster embeddings
    print("[DEBUG] Clustering with HDBSCAN...", flush=True)
    labels, probs, _ = cluster_hdbscan(reduced, min_cluster_size=min_cluster_size, min_samples=min_samples)
    print("[DEBUG] Clustering complete.", flush=True)

    df_unique["cluster_id"] = labels
    df_unique["cluster_confidence"] = probs

    # Map cluster labels back to full dataset
    cluster_map = df_unique[["point_clean", "cluster_id", "cluster_confidence"]].drop_duplicates()
    df_full = df.merge(cluster_map, on="point_clean", how="left")

    # Select and save essential columns
    output_cols = ["document_id", "party", "point", "cluster_id", "cluster_confidence", "speaker", "intervention_id"]
    df_output = df_full[output_cols].copy()
    df_output.to_csv(clustered_file, index=False)

    # Print summary
    print("\n[SUMMARY]")
    print(f"Total rows: {len(df)}")
    print(f"Unique points: {len(df_unique)}")
    print(f"Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
    print(f"Noise points: {(labels == -1).sum()}")
    print(f"\nOutput file:")
    print(f"  - {clustered_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args.input_csv, args.output_dir, args.model_name, args.add_timestamp,
         args.min_cluster_size, args.min_samples, args.n_neighbors, args.min_dist)
