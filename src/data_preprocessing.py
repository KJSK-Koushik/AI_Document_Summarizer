# src/data_preprocessing.py
"""
Data loading and preprocessing utilities for AI_DOCUMENT_SUMMARIZER.

Usage:
    from src.data_preprocessing import (
        load_csv, preprocess_text, load_and_preprocess_df, load_sample_article
    )
"""

import re
from pathlib import Path
from typing import Optional

import pandas as pd
import nltk

# Ensure required NLTK packages are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Basic regex patterns used in cleaning
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_PATTERN = re.compile(r"<.*?>")
MULTI_SPACES = re.compile(r"\s+")
NEWLINE_MULTI = re.compile(r"(\r\n|\r|\n){2,}")
# remove non-printable characters
NON_PRINTABLE = re.compile(r"[^\x20-\x7E\n]")

def load_csv(path: Path | str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        path: Path to CSV file.
        nrows: If provided, read only nrows rows (useful for testing).

    Returns:
        pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, nrows=nrows)

def preprocess_text(text: str, lowercase: bool = True, remove_newlines: bool = True) -> str:
    """
    Clean and normalize text for summarization tasks.

    Steps:
    - Convert to string and strip
    - Remove URLs
    - Remove HTML tags
    - Remove non-printable chars
    - Normalize multiple newlines and spaces
    - lowercase

    Args:
        text: raw text string
        lowercase: whether to lowercase the text
        remove_newlines: if True, collapse newlines to single spaces

    Returns:
        cleaned string
    """
    if text is None:
        return ""

    # ensure str
    s = str(text)   

    # Remove URLs and HTML
    s = URL_PATTERN.sub(" ", s)
    s = HTML_TAG_PATTERN.sub(" ", s)

    # Remove non-printable characters
    s = NON_PRINTABLE.sub(" ", s)

    # Normalize newlines and excessive whitespace
    if remove_newlines:
        # convert runs of newlines to a single space
        s = NEWLINE_MULTI.sub("\n", s)  # limit very long runs of newlines
        s = s.replace("\n", " ")
    s = MULTI_SPACES.sub(" ", s)

    # Trim
    s = s.strip()

    if lowercase:
        s = s.lower()

    return s

def load_and_preprocess_df(
    csv_path: Path | str,
    text_col: str = "article",
    summary_col: Optional[str] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load CSV and preprocess text columns in-place.

    Args:
        csv_path: path to CSV file
        text_col: name of the column containing the article text
        summary_col: name of the column containing reference summaries (optional)
        nrows: optional number of rows to read (useful for quick tests)

    Returns:
        DataFrame with cleaned `text_col` and, if provided, cleaned `summary_col`.
    """
    df = load_csv(csv_path, nrows=nrows)

    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' not found in {csv_path}. Columns: {df.columns.tolist()}")

    # Clean text column
    df[text_col] = df[text_col].fillna("").astype(str).map(preprocess_text)

    if summary_col:
        if summary_col not in df.columns:
            raise KeyError(f"Summary column '{summary_col}' not found in {csv_path}.")
        df[summary_col] = df[summary_col].fillna("").astype(str).map(preprocess_text)

    return df

def load_sample_article(sample_path: Path | str) -> str:
    """
    Read and preprocess the sample article text file.

    Returns:
        cleaned article string
    """
    sample_path = Path(sample_path)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample article not found: {sample_path}")
    raw = sample_path.read_text(encoding="utf-8")
    return preprocess_text(raw, lowercase=False, remove_newlines=False)

# Small CLI-style test when run directly
if __name__ == "__main__":
    # locate project config if available (optional import)
    try:
        from src.config import SAMPLE_ARTICLE, DATA
        sample_path = SAMPLE_ARTICLE
    except Exception:
        sample_path = Path("data/sample_texts/sample_article.txt")

    print("Loading sample article from:", sample_path)
    try:
        article = load_sample_article(sample_path)
        # show a truncated preview
        preview = article[:1000] + ("..." if len(article) > 1000 else "")
        print("\n--- PREPROCESSED SAMPLE PREVIEW (first 1000 chars) ---\n")
        print(preview)
    except Exception as e:
        print("Error while loading sample article:", e)

#python src/data_preprocessing.py