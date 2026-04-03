import os
import argparse
from datetime import datetime
import pandas as pd

from embeddings import Embedder
from clustering import reduce_embeddings, cluster_hdbscan
from representatives import get_cluster_representatives


def load_and_prepare_data(input_csv):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    parser.add_argument("--add_timestamp", action="store_true", help="Add timestamp to output filenames")
    return parser.parse_args()


def main(input_csv, output_dir, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", add_timestamp=False):
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames with optional timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
    timestamp_str = f"_{timestamp}" if timestamp else ""
    
    clustered_unique_file = os.path.join(output_dir, f"clustered_unique_points{timestamp_str}.csv")
    clustered_full_file = os.path.join(output_dir, f"clustered_full_points{timestamp_str}.csv")
    reps_file = os.path.join(output_dir, f"cluster_representatives{timestamp_str}.csv")
    sizes_file = os.path.join(output_dir, f"cluster_sizes{timestamp_str}.csv")

    df = load_and_prepare_data(input_csv)

    # cluster on unique points
    df_unique = df.drop_duplicates(subset=["point_clean"]).copy().reset_index(drop=True)

    embedder = Embedder(model_name=model_name)
    embeddings = embedder.encode(df_unique["point_clean"].tolist())

    reduced = reduce_embeddings(embeddings)
    labels, probs, _ = cluster_hdbscan(reduced)

    df_unique["cluster_id"] = labels
    df_unique["cluster_confidence"] = probs

    # map cluster labels back to full dataset
    cluster_map = df_unique[["point_clean", "cluster_id", "cluster_confidence"]].drop_duplicates()
    df_full = df.merge(cluster_map, on="point_clean", how="left")

    df_unique.to_csv(clustered_unique_file, index=False)
    df_full.to_csv(clustered_full_file, index=False)

    reps = get_cluster_representatives(df_unique, embeddings)
    reps_df = pd.DataFrame(reps)
    reps_df.to_csv(reps_file, index=False)

    cluster_sizes = (
        df_full.groupby("cluster_id")
        .size()
        .reset_index(name="n_points")
        .sort_values("n_points", ascending=False)
    )
    cluster_sizes.to_csv(sizes_file, index=False)

    print("Done.")
    print(f"Total rows: {len(df)}")
    print(f"Unique points: {len(df_unique)}")
    print(f"Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
    print(f"Noise points: {(labels == -1).sum()}")


if __name__ == "__main__":
    args = parse_args()
    main(args.input_csv, args.output_dir, args.model_name, args.add_timestamp)