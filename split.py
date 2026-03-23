import re
import pandas as pd


HEADER_PATTERNS = [
    r"Tweede Kamer",
    r"Eerste Kamer",
    r"\bTK\b\s+\d+",
    r"\bAH\b\s+\d+",
    r"\d{1,2}-\d{1,2}-\d{1,2}",   # e.g. 12-8-37
    r"\b\d+\b(?=\s*$)",           # page number at end
]

PARTIES = [
    "VVD",
    "PvdA",
    "PVV",
    "SP",
    "CDA",
    "D66",
    "GroenLinks",
    "ChristenUnie",
    "SGP",
    "PvdD",
    "50PLUS",
    # add the missing 12th party here if needed
]

PARTY_PATTERN = "|".join(re.escape(p) for p in PARTIES)

SPEAKER_PATTERN = re.compile(
    rf"(De voorzitter|"
    rf"De heer [A-Za-zÀ-ÿ'`\- ]+ \(({PARTY_PATTERN})\)|"
    rf"Mevrouw [A-Za-zÀ-ÿ'`\- ]+ \(({PARTY_PATTERN})\)|"
    rf"Minister [A-Za-zÀ-ÿ'`\- ]+|"
    rf"Staatssecretaris [A-Za-zÀ-ÿ'`\- ]+):"
)


def clean_page_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text)

    # remove hyphenation across line breaks
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # remove recurring header patterns
    for pattern in HEADER_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # remove repeated agenda intro if present
    text = re.sub(
        r"Aan de orde is .*?(?=De voorzitter:|De heer|Mevrouw|Minister|Staatssecretaris|Secretaris|$)",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def join_pages(
    df: pd.DataFrame,
    text_col="foi_bodyText",
    debate_id_col="foi_documentId",
    page_order_col="foi_pageNumber"
) -> pd.DataFrame:
    work = df.copy()

    if page_order_col is not None:
        work = work.sort_values([debate_id_col, page_order_col])

    work[text_col] = work[text_col].apply(clean_page_text)

    joined = (
        work.groupby(debate_id_col, as_index=False)[text_col]
        .apply(lambda x: " ".join(x.astype(str)))
        .rename(columns={text_col: "full_text"})
    )

    joined["full_text"] = (
        joined["full_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return joined


def split_interventions(text: str):
    if not text or pd.isna(text):
        return []

    matches = list(SPEAKER_PATTERN.finditer(text))
    if not matches:
        return []

    interventions = []

    preamble = text[:matches[0].start()].strip()
    if preamble:
        interventions.append({
            "speaker_label": "PREAMBLE",
            "speaker": "PREAMBLE",
            "party": None,
            "speech": preamble
        })

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        chunk = text[start:end].strip()

        colon_idx = chunk.find(":")
        speaker_label = chunk[:colon_idx].strip()
        speech = chunk[colon_idx + 1:].strip()

        if speaker_label == "De voorzitter":
            speaker = "De voorzitter"
            party = None
        else:
            m = re.match(rf"(De heer .+|Mevrouw .+) \(({PARTY_PATTERN})\)$", speaker_label)
            if m:
                speaker = m.group(1).strip()
                party = m.group(2).strip()
            else:
                speaker = speaker_label
                party = None

        if len(speech.split()) == 0:
            continue

        interventions.append({
            "speaker_label": speaker_label,
            "speaker": speaker,
            "party": party,
            "speech": speech
        })

    return interventions


# load data
df_pages = pd.read_csv("rutte_ii.csv")

# sample docs
N_DOCS = 30

sample_doc_ids = (
    df_pages["foi_documentId"]
    .drop_duplicates()
    .sample(N_DOCS, random_state=42)
)

all_interventions = []

for doc_id in sample_doc_ids:
    df_doc = df_pages[df_pages["foi_documentId"] == doc_id].copy()
    df_doc = df_doc.sort_values("foi_pageNumber")

    df_doc["clean_text"] = df_doc["foi_bodyText"].apply(clean_page_text)

    full_text = " ".join(df_doc["clean_text"].tolist())
    full_text = re.sub(r"\s+", " ", full_text).strip()

    interventions = split_interventions(full_text)

    for i, interv in enumerate(interventions, start=1):
        if interv["speaker"] == "PREAMBLE":
            continue

        all_interventions.append({
            "document_id": doc_id,
            "intervention_id": i,
            "speaker_label": interv["speaker_label"],
            "speaker": interv["speaker"],
            "party": interv["party"],
            "speech": interv["speech"]
        })

df_interventions = pd.DataFrame(all_interventions)
df_interventions["n_words"] = df_interventions["speech"].str.split().str.len()
df_interventions["preview"] = df_interventions["speech"].str[:200]

df_interventions.to_csv("intervention_sample_30docs.csv", index=False)
print(f"Saved {len(df_interventions)} interventions.")
