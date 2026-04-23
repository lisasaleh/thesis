"""
Normalize manifesto claims using LLM with local context (surrounding sentences).

Usage - Single party:
    python normalize_manifesto.py \
        --manifesto_dir manifesto \
        --output_dir outputs/manifesto \
        --party GL \
        --model_name /path/to/Qwen2.5-32B-Instruct

Usage - All parties:
    python normalize_manifesto.py \
        --manifesto_dir manifesto \
        --output_dir outputs/manifesto \
        --party all \
        --model_name /path/to/Qwen2.5-32B-Instruct
"""

import os
import json
import argparse
import re
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from llm_utils import LocalLLM
from prompts.normalize_prompt import extract_json_with_basic_repair, validate_normalization_output


MANIFESTO_SYSTEM_PROMPT = """Je bent een deskundige politieke analist gespecialiseerd in beleidsposities en manifestclaims.

Taak: Extract het kernbeleidsstandpunt uit een manifestzin.

Richtlijnen:
- Focus op het BELEIDSSTANDPUNT zelf, feitelijk gesteld vanuit het perspectief van de spreker
- Verwijder retoriek, voorbeelden en voorbehouden
- Maak impliciete standpunten expliciet
- Maximaal 1-2 zinnen
- Geen partijnamen - druk uit als feitelijke claim
- Gebruik "wij" ALLEEN als nodig voor betekenis; gebruik anders passief of directe stellingen
- Voeg geen eigen interpretatie toe
- Gebruik lokale context om nuance te begrijpen, maar focus op de TARGET-zin

Output ONLY als geldige JSON:
{
    "normalized_claim": "Het kernbeleidsstandpunt"
}

Voorbeelden:

Input: "Massief investeringen in hernieuwbare energie zijn essentieel voor duurzaamheid."
Output: "Massieve investeringen in hernieuwbare energie zijn essentieel voor duurzaamheid."

Input: "We moeten het onderwijs versterken."
Output: "Het onderwijs moet worden versterkt."

Input: "Ook wordt de geplande bezuiniging op het passend onderwijs teruggedraaid."
Output: "Bezuinigingen op speciaal onderwijs voor kinderen met beperkingen worden teruggedraaid."

Input: "Plastics zijn slecht voor het milieu. Daarom moeten we ze verbieden."
Output: "Plastics moeten worden verboden omdat ze schadelijk zijn voor het milieu."

"""


def build_manifesto_prompt(sentence: str, local_context: str, cmp_code: int) -> str:
    """Build the normalization prompt for a manifesto sentence."""
    prompt = f"""Beleidsdomein (CMP Code): {cmp_code}

LOKALE CONTEXT (omringende zinnen - lees voor begrip, maar focus op de target-zin):
{local_context}

---

ZIN OM TE NORMALISEREN:
"{sentence}"

---

Extract het kernbeleidsstandpunt uit deze zin. Druk het uit als feitelijke claim vanuit het perspectief van de spreker (geen partijnaam). Gebruik "wij" alleen als nodig. Verwijder retoriek maar behoud de exacte beleidsintentie.
"""
    return prompt.strip()


def get_local_context(df: pd.DataFrame, idx: int, context_window: int = 10, text_col: str = "text") -> str:
    """Get surrounding sentences as context."""
    start_idx = max(0, idx - context_window // 2)
    end_idx = min(len(df), idx + context_window // 2)
    
    context_sentences = []
    for i in range(start_idx, end_idx):
        if i != idx:  # Skip the target sentence
            text = str(df.iloc[i][text_col]).strip()
            if text:
                context_sentences.append(f"- {text}")
    
    return "\n".join(context_sentences) if context_sentences else "[No additional context available]"


def clean_sentence(sentence: str) -> str:
    """Remove leading numbering/lettering (e.g., '9.', 'a)', 'c)' from manifesto text."""
    # Matches patterns like: "9. ", "2) ", "a) ", "c) ", etc.
    cleaned = re.sub(r'^[\d\w]+[\.\)\-\:]\s+', '', sentence.strip())
    return cleaned


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize manifesto claims with LLM.")
    parser.add_argument("--manifesto_dir", type=str, required=True, help="Directory containing manifest_*.csv files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for normalized manifesto.")
    parser.add_argument("--party", type=str, required=True, help="Party name for output filename, or 'all' to process all parties.")
    parser.add_argument("--model_name", type=str, required=True, help="Path or name of 32B model.")
    parser.add_argument("--top_codes", type=int, default=5, help="Number of top CMP codes to normalize.")
    parser.add_argument("--context_window", type=int, default=4, help="Number of surrounding sentences for context.")
    parser.add_argument("--text_col", type=str, default="text", help="Column name for manifesto text.")
    parser.add_argument("--code_col", type=str, default="cmp_code", help="Column name for CMP code.")
    parser.add_argument("--checkpoint_every", type=int, default=50, help="Checkpoint every N sentences.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for LLM processing (higher = faster but more VRAM).")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint.")
    
    return parser.parse_args()


def normalize_single_party(manifesto_csv: str, party: str, output_dir: str, model_name: str, 
                          top_codes: int = 5, context_window: int = 4, 
                          text_col: str = "text", code_col: str = "cmp_code",
                          checkpoint_every: int = 50, resume: bool = False,
                          batch_size: int = 8):
    """Normalize manifesto claims for a single party using batch processing."""
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{party}_manifesto_normalized.csv")
    checkpoint_file = output_csv.replace(".csv", ".checkpoint.json")
    
    # Load manifesto
    print(f"\n[INFO] Loading manifesto from {manifesto_csv}")
    df = pd.read_csv(manifesto_csv)
    
    # Validate columns
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in CSV")
    if code_col not in df.columns:
        raise ValueError(f"Column '{code_col}' not found in CSV")
    
    # Get top N codes
    print(f"[INFO] Finding top {top_codes} most common CMP codes...")
    top_codes_list = df[code_col].value_counts().head(top_codes).index.tolist()
    code_to_rank = {code: idx + 1 for idx, code in enumerate(top_codes_list)}
    
    print(f"[INFO] Top codes (by frequency):")
    for rank, code in enumerate(top_codes_list, 1):
        count = len(df[df[code_col] == code])
        print(f"  {rank}. Code {code}: {count} sentences")
    
    # Filter to top codes
    df_filtered = df[df[code_col].isin(top_codes_list)].reset_index(drop=True)
    print(f"[INFO] Total sentences to normalize: {len(df_filtered)}")
    
    # Load model (only once per party)
    print(f"[DEBUG] Loading model: {model_name}")
    llm = LocalLLM(model_name)
    print(f"[DEBUG] Model loaded")
    
    # Resume from checkpoint if exists
    processed_records = []
    start_idx = 0
    
    if resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_idx = checkpoint.get("processed_count", 0)
            processed_records = checkpoint.get("records", [])
        print(f"[INFO] Resuming from checkpoint: {start_idx}/{len(df_filtered)} processed")
    
    # Normalize each sentence using batch processing
    print(f"[INFO] Starting normalization for party {party} (batch_size={batch_size})...")
    
    batch_prompts = []
    batch_metadata = []
    
    for idx in tqdm(range(start_idx, len(df_filtered)), total=len(df_filtered) - start_idx, desc=party):
        row = df_filtered.iloc[idx]
        
        sentence = clean_sentence(str(row[text_col]))
        code = int(row[code_col])
        
        # Skip empty sentences
        if not sentence:
            continue
        
        # Get local context
        local_context = get_local_context(df_filtered, idx, context_window, text_col)
        
        # Build normalization prompt
        user_prompt = build_manifesto_prompt(sentence, local_context, code)
        
        batch_prompts.append(user_prompt)
        batch_metadata.append({
            "original_sentence": sentence,
            "cmp_code": code,
            "code_rank": code_to_rank.get(code, -1),
            "row_index": idx,
            "sentence_length": len(sentence),
        })
        
        # Process batch when full
        if len(batch_prompts) == batch_size or idx == len(df_filtered) - 1:
            if batch_prompts:
                try:
                    responses = llm.batch_generate(
                        prompts=batch_prompts,
                        system_prompt=MANIFESTO_SYSTEM_PROMPT,
                        max_new_tokens=500,
                        temperature=0.0
                    )
                    
                    # Parse all responses
                    for response, metadata in zip(responses, batch_metadata):
                        try:
                            parsed = extract_json_with_basic_repair(response)
                            normalized_claim = parsed.get("normalized_claim", "").strip()
                            
                            if not normalized_claim:
                                print(f"\n[WARN] Empty normalization for row {metadata['row_index']}: {metadata['original_sentence'][:50]}...")
                                normalized_claim = metadata['original_sentence']
                        except Exception as e:
                            print(f"\n[ERROR] Failed to parse row {metadata['row_index']}: {e}")
                            normalized_claim = metadata['original_sentence']
                        
                        # Store result
                        record = {
                            "original_sentence": metadata['original_sentence'],
                            "normalized_sentence": normalized_claim,
                            "cmp_code": metadata['cmp_code'],
                            "code_rank": metadata['code_rank'],
                            "row_index": metadata['row_index'],
                            "sentence_length": metadata['sentence_length'],
                            "normalized_length": len(normalized_claim),
                        }
                        processed_records.append(record)
                
                except Exception as e:
                    print(f"\n[ERROR] Batch processing failed: {e}")
                    # Fallback: process individually
                    for user_prompt, metadata in zip(batch_prompts, batch_metadata):
                        try:
                            response = llm.generate(
                                prompt=user_prompt,
                                system_prompt=MANIFESTO_SYSTEM_PROMPT,
                                max_new_tokens=500,
                                temperature=0.0
                            )
                            parsed = extract_json_with_basic_repair(response)
                            normalized_claim = parsed.get("normalized_claim", "").strip()
                            if not normalized_claim:
                                normalized_claim = metadata['original_sentence']
                        except Exception as parse_err:
                            print(f"\n[ERROR] Fallback failed for row {metadata['row_index']}: {parse_err}")
                            normalized_claim = metadata['original_sentence']
                        
                        record = {
                            "original_sentence": metadata['original_sentence'],
                            "normalized_sentence": normalized_claim,
                            "cmp_code": metadata['cmp_code'],
                            "code_rank": metadata['code_rank'],
                            "row_index": metadata['row_index'],
                            "sentence_length": metadata['sentence_length'],
                            "normalized_length": len(normalized_claim),
                        }
                        processed_records.append(record)
            
            batch_prompts = []
            batch_metadata = []
            
            # Checkpoint every N sentences
            if len(processed_records) > start_idx and (len(processed_records) - start_idx) % checkpoint_every == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        "processed_count": len(processed_records),
                        "records": processed_records,
                        "timestamp": datetime.now().isoformat()
                    }, f)
    
    # Save final output
    out_df = pd.DataFrame(processed_records)
    out_df = out_df.sort_values(["code_rank", "cmp_code", "row_index"]).reset_index(drop=True)
    out_df.to_csv(output_csv, index=False)
    
    print(f"\n[SUCCESS] Normalized {len(out_df)} manifesto sentences for {party}")
    print(f"[INFO] Output saved to: {output_csv}")
    
    # Cleanup checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    # Print summary by code
    print(f"\n[SUMMARY] Sentences by code rank:")
    for rank in sorted(out_df['code_rank'].unique()):
        code = [c for c, r in code_to_rank.items() if r == rank][0]
        count = len(out_df[out_df['code_rank'] == rank])
        print(f"  Rank {rank} (Code {code}): {count} sentences")


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verify manifesto directory exists
    if not os.path.isdir(args.manifesto_dir):
        raise ValueError(f"Manifesto directory not found: {args.manifesto_dir}")
    
    # Handle --party all
    if args.party.lower() == "all":
        print("[INFO] Processing all parties...")
        
        # Find all manifest_*.csv files
        manifest_files = sorted([f for f in os.listdir(args.manifesto_dir) if f.startswith("manifest_") and f.endswith(".csv")])
        
        if not manifest_files:
            raise ValueError(f"No manifest_*.csv files found in {args.manifesto_dir}")
        
        print(f"[INFO] Found {len(manifest_files)} manifesto files:")
        for f in manifest_files:
            print(f"  - {f}")
        
        # Process each party
        for manifest_file in manifest_files:
            # Extract party name: manifest_GL.csv -> GL
            party_name = manifest_file.replace("manifest_", "").replace(".csv", "").upper()
            manifesto_path = os.path.join(args.manifesto_dir, manifest_file)
            
            try:
                normalize_single_party(
                    manifesto_csv=manifesto_path,
                    party=party_name,
                    output_dir=args.output_dir,
                    model_name=args.model_name,
                    top_codes=args.top_codes,
                    context_window=args.context_window,
                    text_col=args.text_col,
                    code_col=args.code_col,
                    checkpoint_every=args.checkpoint_every,
                    batch_size=args.batch_size,
                    resume=args.resume
                )
            except Exception as e:
                print(f"\n[ERROR] Failed to process party {party_name}: {e}")
                continue
        
        print(f"\n[INFO] Completed processing all parties!")
    
    else:
        # Single party mode
        manifesto_file = f"manifest_{args.party.lower()}.csv"
        manifesto_path = os.path.join(args.manifesto_dir, manifesto_file)
        
        if not os.path.exists(manifesto_path):
            raise ValueError(f"Manifesto file not found: {manifesto_path}")
        
        normalize_single_party(
            manifesto_csv=manifesto_path,
            party=args.party.upper(),
            output_dir=args.output_dir,
            model_name=args.model_name,
            top_codes=args.top_codes,
            context_window=args.context_window,
            text_col=args.text_col,
            code_col=args.code_col,
            checkpoint_every=args.checkpoint_every,
            batch_size=args.batch_size,
            resume=args.resume
        )


if __name__ == "__main__":
    main()
