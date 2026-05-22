# AI Document Summarizer

Legacy AI-powered document summarization project built with Hugging Face Transformers, PyTorch, and Gradio.

This repository contains the source code for training, running, and evaluating an abstractive document summarizer. The current implementation uses a fine-tuned sequence-to-sequence Transformer model, with support for text, PDF, and DOCX inputs through a local Gradio interface.

## Current Architecture

The project is based on a classic encoder-decoder summarization pipeline:

```text
Input document
-> text extraction
-> preprocessing / cleaning
-> tokenization
-> seq2seq summarization model
-> generated summary
-> optional ROUGE evaluation
```

The saved legacy model was trained as a T5-style summarizer. Some configuration files mention BART as a possible default, but the local saved model metadata identifies the actual trained checkpoint as `t5-small`.

## Project Structure

```text
.
├── app.py                         # Gradio web app
├── main.py                        # Empty legacy entry point
├── requirements.txt               # Python dependencies
├── data/
│   └── sample_texts/
│       └── sample_article.txt     # Small sample input
├── docs/
│   └── architecture_overview.md   # Legacy architecture notes
├── src/
│   ├── config.py                  # Paths and model/training config
│   ├── data_preprocessing.py      # CSV loading and text cleaning
│   ├── evaluate.py                # ROUGE evaluation
│   ├── generate_predictions.py    # Batch prediction generation
│   ├── model.py                   # Model loading and summarization helpers
│   ├── summarize.py               # CLI summarization
│   └── train.py                   # Fine-tuning loop
├── tools/
│   └── md_to_pdf.py               # Markdown-to-PDF helper
└── utils/
    └── file_utils.py              # PDF/DOCX/TXT extraction helpers
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Gradio App

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:7861
```

The app supports:

- PDF upload
- DOCX upload
- TXT upload
- pasted text input
- beam search and length controls
- downloadable summary output

## CLI Usage

Summarize the sample article:

```bash
python -m src.summarize data/sample_texts/sample_article.txt
```

Generate predictions for a CSV file:

```bash
python -m src.generate_predictions data/test.csv 50
```

Evaluate predictions with ROUGE:

```bash
python -m src.evaluate data/test.csv outputs/results/predictions_file.csv
```

## Training

The legacy training script expects CSV files with:

- `article`
- `highlights`

Example:

```bash
python -m src.train ^
  --train_file data/train.csv ^
  --val_file data/validation.csv ^
  --model_name t5-small ^
  --output_dir outputs/model/best_model ^
  --num_train_epochs 1 ^
  --per_device_train_batch_size 2 ^
  --max_samples 200
```

## Files Not Tracked in Git

Large generated files are intentionally excluded from this repository:

- `.venv/`
- `outputs/`
- `data/*.csv`
- model checkpoints
- optimizer states
- generated result files
- demo video files

This keeps the repository lightweight and avoids GitHub file size limits.

If you need to reproduce the full legacy environment, restore the dataset CSVs and model checkpoint files locally using the same folder layout expected by `src/config.py`.

## Known Limitations

This is the preserved legacy version of the project. Current limitations include:

- short-context model behavior
- hard truncation around legacy model input limits
- naive character-based chunking in the UI
- no citation grounding
- no streaming generation
- no hallucination/faithfulness evaluation
- ROUGE-only quality checks
- no production API layer or job queue

## Modernization Direction

The next planned upgrade is to migrate this into a production-grade document summarization system with:

- layout-aware document ingestion
- semantic chunking
- anchor-span extraction
- grounded abstractive summaries
- structured JSON outputs
- streaming responses
- automated groundedness evaluation
- provider-agnostic LLM routing
