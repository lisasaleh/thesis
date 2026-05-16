import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "outputs/samples/VVD_cmp_1_normalized.csv"
MANIFEST_FILE = "outputs/cmp_manifest.csv"
OUTPUT_FILE = "outputs/samples/VVD_cmp_1_labeled.csv"

MODEL_NAME = (
    "manifesto-project/"
    "manifestoberta-xlm-roberta-56policy-topics-sentence-2024-1-1"
)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model.eval()

# ============================================================
# HELPERS
# ============================================================


def get_party_and_rank_from_filename(filename: str):
    """
    Example:
        VVD cmp_1.csv

    Returns:
        ("VVD", 1)
    """

    # operate on the basename (remove path and extension)
    base = os.path.basename(filename)
    base = os.path.splitext(base)[0]

    match = re.search(
        r"(.+?)\s*cmp[_\s-]*(\d+)",
        base,
        re.IGNORECASE
    )

    if not match:
        raise ValueError(
            f"Could not parse party/rank from filename: {filename}"
        )

    # remove trailing separators from the extracted party name
    party = re.sub(r"[_\s-]+$", "", match.group(1).strip())
    rank = int(match.group(2))

    return party, rank



def get_target_cmp_code(filename: str, manifest_df: pd.DataFrame):
    """
    Example:
        VVD_cmp_1.csv

    Looks up:
        party == VVD
        code_1

    Returns:
        605
    """

    party, rank = get_party_and_rank_from_filename(filename)

    row = manifest_df.loc[
        manifest_df["party"].astype(str).str.strip().eq(party)
    ]

    if row.empty:
        raise ValueError(f"Party {party!r} not found")

    code_col = f"code_{rank}"

    if code_col not in manifest_df.columns:
        raise ValueError(f"Column {code_col!r} not found")

    target_code = str(row.iloc[0][code_col]).strip()

    return party, rank, target_code



def normalize_cmp_code(label: str):
    """
    ManifestoBERT sometimes returns labels like:

        per605 or 605 - Law and Order: Positive

    Convert to:

        605
    """

    label = str(label)
    label = label.strip()
    # extract only the leading numeric part
    match = re.match(r"^(\d+)", label)
    if match:
        return match.group(1)
    return ""


def get_confidence_for_code(text: str, target_code: str):
    """
    Get the confidence score for a specific CMP code.
    
    Returns:
        confidence (float)
    """
    
    text = str(text)
    target_code = str(target_code).strip()
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Find the target code in id2label mapping
    target_id = None
    for id_val, label_str in model.config.id2label.items():
        normalized = normalize_cmp_code(label_str)
        if normalized == target_code:
            target_id = id_val
            break
    
    if target_id is None:
        return 0.0
    
    confidence = probs[0, target_id].item()
    return confidence


# ============================================================
# CLASSIFICATION
# ============================================================


def classify_point(text: str):
    """
    Returns:
        predicted_code
        confidence
    """

    text = str(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Highest probability class
    pred_id = probs.argmax(dim=-1).item()

    # Probability of predicted class
    confidence = probs[0, pred_id].item()

    # Convert class id to CMP label
    label = model.config.id2label[pred_id]

    label = normalize_cmp_code(label)

    return label, confidence


# ============================================================
# MAIN
# ============================================================

print("Loading CSV files...")

manifest_df = pd.read_csv(MANIFEST_FILE)
df = pd.read_csv(INPUT_FILE)

party, cmp_rank, target_cmp_code = get_target_cmp_code(
    INPUT_FILE,
    manifest_df
)

print(f"Party: {party}")
print(f"CMP Rank: {cmp_rank}")
print(f"Target CMP Code: {target_cmp_code}")

# Run classifier
print("Running classification...")

results = df["point"].fillna("").apply(classify_point)

# Store predictions
# results[i] = (label, confidence)

df["predicted_cmp_code"] = results.apply(lambda x: x[0])
df["cmp_confidence"] = results.apply(lambda x: x[1])

# Metadata

df["target_cmp_code"] = target_cmp_code
df["party_from_filename"] = party
df["cmp_rank_from_filename"] = cmp_rank

# Get confidence for target CMP code
print("Computing target code confidence...")
df["target_code_confidence"] = df["point"].fillna("").apply(
    lambda x: get_confidence_for_code(x, target_cmp_code)
)

# Exact match with target CMP code

df["matches_target_cmp"] = (
    df["predicted_cmp_code"].astype(str)
    ==
    df["target_cmp_code"].astype(str)
)

# Optional: sort by confidence

df = df.sort_values(
    by="cmp_confidence",
    ascending=False
)

print(df[[
    "point",
    "predicted_cmp_code",
    "cmp_confidence",
    "target_code_confidence",
    "matches_target_cmp"
]].head())

# Save

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved output to: {OUTPUT_FILE}")
