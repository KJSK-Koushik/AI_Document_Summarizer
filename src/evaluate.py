# src/evaluate.py
"""
Evaluate generated summaries against reference summaries using ROUGE metrics.

Usage:
------
python -m src.evaluate data/test.csv outputs/results/predictions_*.csv

If you only want to test ROUGE on your validation set after fine-tuning,
pass your validation CSV and generated summaries file.
"""

import sys
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from rouge_score import rouge_scorer
from statistics import mean

from src.config import RESULTS_DIR

def load_data(csv_path: Path, text_col="article", summary_col="highlights"):
    """Load the dataset (with reference summaries)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    if summary_col not in df.columns:
        raise KeyError(f"Column '{summary_col}' not found in dataset.")
    return df

def load_predictions(pred_path: Path):
    """Load model-generated summaries (from summarize.py CSV)."""
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    df = pd.read_csv(pred_path)
    if "summary" not in df.columns:
        raise KeyError("Predictions CSV must contain a 'summary' column.")
    return df

def compute_rouge(reference_texts, generated_texts):
    """
    Compute mean ROUGE-1, ROUGE-2, ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for ref, gen in zip(reference_texts, generated_texts):
        score = scorer.score(ref, gen)
        for key in scores:
            scores[key].append(score[key].fmeasure)

    avg_scores = {k: round(mean(v) * 100, 2) for k, v in scores.items()}
    return avg_scores

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m src.evaluate <reference_csv> <predictions_csv>")
        sys.exit(1)

    ref_csv = Path(sys.argv[1])
    pred_csv = Path(sys.argv[2])

    print(f"Loading reference dataset: {ref_csv}")
    print(f"Loading predictions: {pred_csv}")

    ref_df = load_data(ref_csv)
    pred_df = load_predictions(pred_csv)

    # Align sizes if needed
    n = min(len(ref_df), len(pred_df))
    ref_texts = ref_df["highlights"].head(n).fillna("").tolist()
    gen_texts = pred_df["summary"].head(n).fillna("").tolist()

    print(f"Evaluating {n} summaries...")
    scores = compute_rouge(ref_texts, gen_texts)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"rouge_scores_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(scores, f, indent=2)

    print("\n✅ ROUGE Evaluation Complete!")
    for k, v in scores.items():
        print(f"{k}: {v}")
    print(f"\nScores saved to: {out_path}")

if __name__ == "__main__":
    main()
