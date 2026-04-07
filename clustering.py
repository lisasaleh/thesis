import umap
import hdbscan


def reduce_embeddings(embeddings, n_neighbors=8, n_components=10, min_dist=0.0, metric="cosine", random_state=42):
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
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(reduced_embeddings)
    probabilities = clusterer.probabilities_
    return labels, probabilities, clusterer