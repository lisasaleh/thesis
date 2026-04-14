import argparse
import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class CentroidSelector:
    """
    Cluster representative selector using centroid-based approach.

    For each cluster:
    1. Embed all claims using sentence transformer
    2. Compute cluster centroid as mean embedding
    3. Find claim with highest cosine similarity to centroid
    4. Select that claim as representative
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: str = None,
    ):
        self.model_name = model_name
        self.device = device
        print(f"[DEBUG] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"[DEBUG] Model loaded successfully")

    def select_representative(
        self,
        cluster_df: pd.DataFrame,
        text_col: str,
        id_col: str = None,
    ) -> Dict[str, Any]:
        """
        Select representative for one cluster using centroid method.
        """
        cluster_df = cluster_df.reset_index(drop=True).copy()
        texts = cluster_df[text_col].fillna("").astype(str).tolist()
        n = len(texts)

        if n == 0:
            raise ValueError("Empty cluster received.")

        if n == 1:
            row = cluster_df.iloc[0]
            return {
                "representative_idx": 0,
                "representative_text": row[text_col],
                "representative_uid": row[id_col] if id_col and id_col in cluster_df.columns else None,
                "cluster_size": 1,
                "centroid_similarity": 1.0,
                "representative_quality": "singleton",
            }

        # Embed all texts
        embeddings = self.model.encode(texts)  # shape: (n, embedding_dim)
        
        # Compute centroid
        centroid = np.mean(embeddings, axis=0)  # shape: (embedding_dim,)
        
        # Compute cosine similarity from each embedding to centroid
        similarities = cosine_similarity(
            embeddings.reshape(n, -1),
            centroid.reshape(1, -1)
        ).flatten()  # shape: (n,)
        
        # Find the index with max similarity
        best_idx = np.argmax(similarities)
        best_similarity = float(similarities[best_idx])
        
        row = cluster_df.iloc[best_idx]
        return {
            "representative_idx": int(best_idx),
            "representative_text": row[text_col],
            "representative_uid": row[id_col] if id_col and id_col in cluster_df.columns else None,
            "cluster_size": int(n),
            "centroid_similarity": round(best_similarity, 6),
            "representative_quality": "centroid",
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Select one representative claim per cluster using centroid-based method.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to clustered CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for selection results.")
    parser.add_argument("--party", type=str, required=True, help="Party name for output filename.")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", help="Sentence transformer model for embeddings.")
    parser.add_argument("--cluster_col", type=str, default="cluster_id", help="Cluster column.")
    parser.add_argument("--text_col", type=str, default="point", help="Text column containing normalized claims.")
    parser.add_argument("--id_col", type=str, default="point_uid", help="Optional unique claim id column.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda. Default: auto.")
    parser.add_argument(
        "--keep_cluster_metadata",
        action="store_true",
        help="If set, copy selected row metadata into output."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename based on party
    output_csv = os.path.join(args.output_dir, f"{args.party}_selection.csv")

    df = pd.read_csv(args.input_csv)

    required_cols = [args.cluster_col, args.text_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input CSV.")

    selector = CentroidSelector(
        model_name=args.model_name,
        device=args.device,
    )

    results = []

    grouped = df.groupby(args.cluster_col, sort=True)

    for cluster_id, cluster_df in grouped:
        # Drop empty texts inside cluster
        cluster_df = cluster_df[cluster_df[args.text_col].notna()].copy()
        cluster_df[args.text_col] = cluster_df[args.text_col].astype(str).str.strip()
        cluster_df = cluster_df[cluster_df[args.text_col] != ""].copy()

        if len(cluster_df) == 0:
            continue

        rep = selector.select_representative(
            cluster_df=cluster_df,
            text_col=args.text_col,
            id_col=args.id_col if args.id_col in cluster_df.columns else None,
        )

        out_row = {
            args.cluster_col: cluster_id,
            "representative_text": rep["representative_text"],
            "representative_uid": rep["representative_uid"],
            "cluster_size": rep["cluster_size"],
            "centroid_similarity": rep["centroid_similarity"],
            "representative_quality": rep["representative_quality"],
        }

        if args.keep_cluster_metadata:
            selected_idx = rep["representative_idx"]
            selected_row = cluster_df.reset_index(drop=True).iloc[selected_idx].to_dict()

            # avoid overwriting output keys
            for key, value in selected_row.items():
                if key not in out_row:
                    out_row[key] = value
                else:
                    out_row[f"selected_{key}"] = value

        results.append(out_row)

    out_df = pd.DataFrame(results)

    # Optional ordering
    if args.cluster_col in out_df.columns:
        out_df = out_df.sort_values(args.cluster_col).reset_index(drop=True)

    out_df.to_csv(output_csv, index=False)

    print(f"Saved {len(out_df)} cluster representatives to: {output_csv}")


if __name__ == "__main__":
    main()