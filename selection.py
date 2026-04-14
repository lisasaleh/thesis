import argparse
import itertools
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class PairScore:
    i: int
    j: int
    prob: float
    pred: int


class KPMSelector:
    """
    Pairwise cluster representative selector.

    For each cluster:
    1. score all unordered pairs of claims with a sequence classification model
    2. count positive matches per claim
    3. select the claim with the highest support as the representative
    """

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.5,
        batch_size: int = 16,
        max_length: int = 256,
        device: str = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_pairs(self, texts_a: List[str], texts_b: List[str]) -> List[float]:
        """
        Returns P(match) for each pair.

        Supports:
        - binary classifier with 1 logit -> sigmoid
        - 2+ label classifier -> softmax, uses last label as positive class
        """
        probs = []

        for start in range(0, len(texts_a), self.batch_size):
            batch_a = texts_a[start:start + self.batch_size]
            batch_b = texts_b[start:start + self.batch_size]

            enc = self.tokenizer(
                batch_a,
                batch_b,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            outputs = self.model(**enc)
            logits = outputs.logits

            if logits.shape[-1] == 1:
                batch_probs = torch.sigmoid(logits.squeeze(-1))
            else:
                batch_probs = torch.softmax(logits, dim=-1)[:, -1]

            probs.extend(batch_probs.detach().cpu().tolist())

        return probs

    def select_representative(
        self,
        cluster_df: pd.DataFrame,
        text_col: str,
        id_col: str = None,
    ) -> Dict[str, Any]:
        """
        Select representative for one cluster.
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
                "best_match_count": 0,
                "best_match_sum": 0.0,
                "avg_match_prob": 0.0,
                "representative_quality": "singleton",
            }

        pair_indices = list(itertools.combinations(range(n), 2))
        texts_a = [texts[i] for i, _ in pair_indices]
        texts_b = [texts[j] for _, j in pair_indices]

        pair_probs = self.score_pairs(texts_a, texts_b)

        # symmetric support matrix
        support_counts = np.zeros(n, dtype=int)
        support_sums = np.zeros(n, dtype=float)
        all_pair_probs_per_claim = [[] for _ in range(n)]

        scored_pairs: List[PairScore] = []
        for (i, j), prob in zip(pair_indices, pair_probs):
            pred = int(prob >= self.threshold)
            scored_pairs.append(PairScore(i=i, j=j, prob=prob, pred=pred))

            if pred == 1:
                support_counts[i] += 1
                support_counts[j] += 1

            support_sums[i] += prob
            support_sums[j] += prob

            all_pair_probs_per_claim[i].append(prob)
            all_pair_probs_per_claim[j].append(prob)

        avg_probs = np.array(
            [
                float(np.mean(p_list)) if len(p_list) > 0 else 0.0
                for p_list in all_pair_probs_per_claim
            ]
        )

        # Ranking:
        # 1. most positive matches
        # 2. highest summed probability
        # 3. highest average probability
        # 4. shortest claim (slightly favors concise key points)
        # 5. first occurrence
        lengths = np.array([len(t) for t in texts])

        candidate_order = sorted(
            range(n),
            key=lambda idx: (
                support_counts[idx],
                support_sums[idx],
                avg_probs[idx],
                -lengths[idx],   # shorter is better -> invert in ascending sort later? no, see reverse=True
                -idx
            ),
            reverse=True
        )
        best_idx = candidate_order[0]

        # Diagnostic label for cluster quality
        max_possible_matches = n - 1
        best_count = int(support_counts[best_idx])
        best_sum = float(support_sums[best_idx])
        best_avg = float(avg_probs[best_idx])

        # Heuristic quality label
        # strong: representative matches at least half of the cluster
        # weak: best candidate exists but has limited support
        if best_count >= math.ceil(max_possible_matches / 2):
            quality = "strong"
        else:
            quality = "weak"

        row = cluster_df.iloc[best_idx]
        return {
            "representative_idx": int(best_idx),
            "representative_text": row[text_col],
            "representative_uid": row[id_col] if id_col and id_col in cluster_df.columns else None,
            "cluster_size": int(n),
            "best_match_count": best_count,
            "best_match_sum": round(best_sum, 6),
            "avg_match_prob": round(best_avg, 6),
            "representative_quality": quality,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Select one representative claim per cluster using pairwise KPM scoring.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to clustered CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for selection results.")
    parser.add_argument("--party", type=str, required=True, help="Party name for output filename.")
    parser.add_argument("--model_name", type=str, required=True, help="HF model for pairwise match classification.")
    parser.add_argument("--cluster_col", type=str, default="cluster_id", help="Cluster column.")
    parser.add_argument("--text_col", type=str, default="point", help="Text column containing normalized claims.")
    parser.add_argument("--id_col", type=str, default="point_uid", help="Optional unique claim id column.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for positive match.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for model inference.")
    parser.add_argument("--max_length", type=int, default=256, help="Max tokenizer length.")
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

    selector = KPMSelector(
        model_name=args.model_name,
        threshold=args.threshold,
        batch_size=args.batch_size,
        max_length=args.max_length,
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
            "best_match_count": rep["best_match_count"],
            "best_match_sum": rep["best_match_sum"],
            "avg_match_prob": rep["avg_match_prob"],
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