# src/generate_predictions.py
"""
Generate summaries for rows in a CSV and save to outputs/results/predictions_<timestamp>.csv
Usage:
    python -m src.generate_predictions data/test.csv 50
"""
import sys
import csv
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.model import load_model_and_tokenizer, summarize_text
from src.config import RESULTS_DIR

def generate_for_csv(csv_path: Path, text_col="article", summary_col="highlights", nrows=None):
    df = pd.read_csv(csv_path, nrows=nrows)
    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' not found. Available: {df.columns.tolist()}")
    if summary_col not in df.columns:
        raise KeyError(f"Summary column '{summary_col}' not found. Available: {df.columns.tolist()}")

    model, tokenizer = load_model_and_tokenizer()
    results = []
    for idx, row in df.iterrows():
        article = str(row[text_col])
        ref = str(row[summary_col])
        print(f"Summarizing row {idx+1}/{len(df)}...")
        summary = summarize_text(article, model, tokenizer)
        results.append({"id": idx, "article": article, "reference": ref, "summary": summary})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"predictions_{ts}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id","article","reference","summary"])
        writer.writeheader()
        writer.writerows(results)
    print("✅ Saved predictions to:", out_path)
    return out_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.generate_predictions <csv_path> [nrows]")
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    nrows = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    generate_for_csv(csv_path, nrows=nrows)
