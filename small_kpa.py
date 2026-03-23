import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ollama

# =========================
# Config
# =========================

INPUT_CSV = "intervention_sample_30docs.csv"
DOC_ID = "nl.oorg10002.2b.2015.20142015-60-10.doc.1"

OLLAMA_MODEL = "llama3.2:3b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MIN_WORDS = 12
DROP_CHAIR = True
N_CLUSTERS = 8

OUTPUT_CLAIMS_CSV = "local_kpa_claims.csv"
OUTPUT_KPS_CSV = "local_kpa_keypoints.csv"

# =========================
# Load data
# =========================

df = pd.read_csv(INPUT_CSV)

df_doc = df[df["document_id"] == DOC_ID].copy()
if df_doc.empty:
    raise ValueError(f"No rows found for document_id={DOC_ID}")

if "n_words" not in df_doc.columns:
    df_doc["n_words"] = df_doc["speech"].fillna("").str.split().str.len()

df_doc = df_doc[df_doc["n_words"] >= MIN_WORDS].copy()

if DROP_CHAIR:
    df_doc = df_doc[df_doc["speaker"] != "De voorzitter"].copy()

df_doc["speech"] = df_doc["speech"].fillna("").astype(str).str.strip()
df_doc = df_doc[df_doc["speech"] != ""].reset_index(drop=True)

print(f"Document: {DOC_ID}")
print(f"Interventions after filtering: {len(df_doc)}")

# =========================
# Claim extraction with Ollama
# =========================

def build_prompt(speech: str) -> str:
    return f"""
Extract the main political claim from this Dutch parliamentary intervention.

Rules:
- Return exactly one concise sentence.
- Focus on the substantive political argument, proposal, criticism, or position.
- Ignore greetings, procedural language, and rhetorical filler.
- Do not explain.
- Do not use bullet points.

Intervention:
{speech}
""".strip()

def extract_claim_local(speech: str) -> str:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": build_prompt(speech)
            }
        ]
    )
    return response["message"]["content"].strip()

tqdm.pandas(desc="Extracting claims")
df_doc["claim"] = df_doc["speech"].progress_apply(extract_claim_local)

df_doc["claim"] = df_doc["claim"].fillna("").str.strip()
df_doc = df_doc[df_doc["claim"] != ""].reset_index(drop=True)

# =========================
# Embeddings
# =========================

embedder = SentenceTransformer(EMBED_MODEL)
embeddings = embedder.encode(
    df_doc["claim"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

df_doc["embedding"] = list(embeddings)

# =========================
# Clustering
# =========================

if len(df_doc) < N_CLUSTERS:
    raise ValueError(
        f"Only {len(df_doc)} claims left, fewer than N_CLUSTERS={N_CLUSTERS}"
    )

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df_doc["cluster"] = kmeans.fit_predict(embeddings)

# =========================
# Representative claim per cluster
# =========================

def representative_claim(cluster_df: pd.DataFrame) -> str:
    X = np.vstack(cluster_df["embedding"].to_list())
    sims = cosine_similarity(X, X)
    centrality = sims.mean(axis=1)
    idx = int(np.argmax(centrality))
    return cluster_df.iloc[idx]["claim"]

keypoint_rows = []

for cluster_id in sorted(df_doc["cluster"].unique()):
    cluster_df = df_doc[df_doc["cluster"] == cluster_id].copy()
    kp = representative_claim(cluster_df)

    keypoint_rows.append({
        "document_id": DOC_ID,
        "cluster": int(cluster_id),
        "cluster_size": int(len(cluster_df)),
        "key_point": kp
    })

df_kp = pd.DataFrame(keypoint_rows).sort_values(
    ["cluster_size", "cluster"],
    ascending=[False, True]
).reset_index(drop=True)

# =========================
# Save outputs
# =========================

df_doc["preview"] = df_doc["speech"].str[:200]
df_doc["n_words"] = df_doc["speech"].str.split().str.len()

save_cols = [c for c in df_doc.columns if c != "embedding"]
df_doc[save_cols].to_csv(OUTPUT_CLAIMS_CSV, index=False)
df_kp.to_csv(OUTPUT_KPS_CSV, index=False)

print(f"Saved claims to {OUTPUT_CLAIMS_CSV}")
print(f"Saved key points to {OUTPUT_KPS_CSV}")

print("\nKey Points:")
for _, row in df_kp.iterrows():
    print(f"- [cluster {row['cluster']}, n={row['cluster_size']}] {row['key_point']}")