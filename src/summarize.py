# src/summarize.py
"""
Summarize any text file(s) using the pretrained or fine-tuned model.

Usage Examples:
---------------
1. Summarize a single file:
   python -m src.summarize data/sample_texts/sample_article.txt

2. Summarize all .txt files in a folder:
   python -m src.summarize data/sample_texts/

Results are saved in: outputs/results/predictions.csv
"""

import sys
import csv
from pathlib import Path
from datetime import datetime

from src.config import MODEL_DIR, RESULTS_DIR, SAMPLE_ARTICLE
from src.model import load_model_and_tokenizer, summarize_text


def summarize_file(input_path: Path, model, tokenizer):
    """Summarize a single text file and return summary string."""
    text = input_path.read_text(encoding="utf-8")
    summary = summarize_text(text, model, tokenizer)
    return summary


def summarize_folder(folder_path: Path, model, tokenizer):
    """Summarize all .txt files in a folder."""
    summaries = []
    for file in folder_path.glob("*.txt"):
        print(f"\nSummarizing: {file.name}")
        summary = summarize_file(file, model, tokenizer)
        summaries.append({"filename": file.name, "summary": summary})
    return summaries


def save_results_to_csv(results, output_path: Path):
    """Save summaries to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["filename", "summary"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n Results saved to: {output_path}")


def main():
    # Handle command-line args
    if len(sys.argv) > 1:
        input_arg = Path(sys.argv[1])
    else:
        # Default: summarize sample article
        input_arg = SAMPLE_ARTICLE

    print(f"Input path: {input_arg}")
    if not input_arg.exists():
        print(f" Error: Path not found: {input_arg}")
        sys.exit(1)

    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()  # load from pretrained name or HF cache


    results = []

    if input_arg.is_file():
        print(f"\nSummarizing file: {input_arg.name}")
        summary = summarize_file(input_arg, model, tokenizer)
        results.append({"filename": input_arg.name, "summary": summary})
        print("\n--- SUMMARY ---\n")
        print(summary)
    elif input_arg.is_dir():
        results = summarize_folder(input_arg, model, tokenizer)
    else:
        print(" Input must be a file or folder.")
        sys.exit(1)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = RESULTS_DIR / f"predictions_{timestamp}.csv"
    save_results_to_csv(results, output_csv)


if __name__ == "__main__":
    main()
