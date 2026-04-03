import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_cluster_representatives(df, embeddings, text_col="point_clean", cluster_col="cluster_id", top_k=5):
    representatives = []

    for cluster_id in sorted(c for c in df[cluster_col].unique() if c != -1):
        cluster_idx = np.where(df[cluster_col].values == cluster_id)[0]
        cluster_emb = embeddings[cluster_idx]

        centroid = cluster_emb.mean(axis=0, keepdims=True)
        sims = cosine_similarity(cluster_emb, centroid).ravel()
        best_local = cluster_idx[np.argsort(-sims)[:top_k]]

        for rank, idx in enumerate(best_local, start=1):
            row = df.iloc[idx]
            representatives.append({
                "cluster_id": cluster_id,
                "rank": rank,
                "point_uid": row["point_uid"],
                "party": row["party"],
                "speaker": row["speaker"],
                "point_clean": row[text_col],
                "quote": row["quote"]
            })

    return representatives