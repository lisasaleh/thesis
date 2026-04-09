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
    """Main clustering pipeline - clusters per document."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with optional timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
    timestamp_str = f"_{timestamp}" if timestamp else ""
    
    clustered_file = os.path.join(output_dir, f"clustered_points{timestamp_str}.csv")

    # Load and prepare data
    df = load_and_prepare_data(input_csv)
    
    # Initialize embedder once
    print("[DEBUG] Loading embedding model...", flush=True)
    embedder = Embedder(model_name=model_name)
    print("[DEBUG] Embeddings model loaded.", flush=True)
    
    # Store results per document
    all_results = []
    
    # Cluster per document
    unique_docs = df["document_id"].unique()
    print(f"[DEBUG] Clustering {len(unique_docs)} document(s)...", flush=True)
    
    for doc_id in unique_docs:
        df_doc = df[df["document_id"] == doc_id].copy()
        
        # Get unique points in this document
        df_doc_unique = df_doc.drop_duplicates(subset=["point_clean"]).copy().reset_index(drop=True)
        
        if len(df_doc_unique) == 0:
            continue
        
        print(f"\n[DEBUG] Processing document: {doc_id} ({len(df_doc_unique)} unique points)")
        
        # Generate embeddings for this document's points
        embeddings = embedder.encode(df_doc_unique["point_clean"].tolist())
        
        # Reduce dimensionality
        reduced = reduce_embeddings(embeddings, n_neighbors=n_neighbors, min_dist=min_dist)
        
        # Cluster embeddings
        labels, probs, _ = cluster_hdbscan(reduced, min_cluster_size=min_cluster_size, min_samples=min_samples)
        
        # Assign cluster IDs with document prefix to ensure global uniqueness
        df_doc_unique["cluster_id"] = [f"{doc_id}_cluster_{label}" if label != -1 else -1 for label in labels]
        df_doc_unique["cluster_confidence"] = probs
        
        print(f"  Clusters: {len(set(label for label in labels if label != -1))} | Noise: {(labels == -1).sum()}")
        
        # Map back to full document data
        cluster_map = df_doc_unique[["point_clean", "cluster_id", "cluster_confidence"]].drop_duplicates()
        df_doc_clustered = df_doc.merge(cluster_map, on="point_clean", how="left")
        
        all_results.append(df_doc_clustered)
    
    # Combine all documents
    df_output_full = pd.concat(all_results, ignore_index=True)
    
    # Select and save essential columns
    output_cols = ["document_id", "party", "point", "cluster_id", "cluster_confidence", "speaker", "intervention_id"]
    df_output = df_output_full[output_cols].copy()
    df_output.to_csv(clustered_file, index=False)

    # Print summary
    print("\n[SUMMARY]")
    print(f"Total rows: {len(df)}")
    print(f"Documents processed: {len(unique_docs)}")
    total_clusters = len(set(cid for cid in df_output["cluster_id"] if cid != -1))
    total_noise = (df_output["cluster_id"] == -1).sum()
    print(f"Total clusters (all docs): {total_clusters}")
    print(f"Total noise points: {total_noise}")
    print(f"\nOutput file:")
    print(f"  - {clustered_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args.input_csv, args.output_dir, args.model_name, args.add_timestamp,
         args.min_cluster_size, args.min_samples, args.n_neighbors, args.min_dist)
