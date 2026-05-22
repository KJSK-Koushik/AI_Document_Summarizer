# AI Document Summarizer — Architecture Overview

## 📌 Big Picture

**Primary objective:** This repository provides a complete pipeline to train, run, and evaluate an abstractive document summarization system based on pre-trained sequence‑to‑sequence Transformer models (e.g., BART/T5/Pegasus). The code supports: data preprocessing, fine‑tuning with Hugging Face Trainer, generating summaries (batched or interactive), evaluating with ROUGE, and a Gradio app for local inference.

---

## 🔧 Model Architecture & Libraries

**Models used:**
- Primary default model: `facebook/bart-large-cnn` (configurable via `src/config.py`)
- The code supports other seq2seq models (e.g., `t5-small`, `google/pegasus-xsum`) since it uses Hugging Face Transformers `AutoModelForSeq2SeqLM` and `AutoTokenizer`.

**Libraries / frameworks:**
- transformers (Hugging Face) — model + tokenizer + pipeline + Trainer
- datasets — light integration for dataset objects used in training
- evaluate / rouge-score — for ROUGE metric computation
- PyTorch — backend for model execution
- gradio — lightweight UI for inference
- pandas, pdfplumber, python-docx — data handling and file extraction utilities

**Why these choices?**
- BART/T5/Pegasus are proven, pre-trained seq2seq models for abstractive summarization; they provide strong generation and are easy to fine-tune via HF Trainer.
- Hugging Face ecosystem (transformers, datasets, evaluate) greatly simplifies model loading, tokenization, training loops, and evaluation.
- Gradio provides a fast way to build a shareable local UI.

---

## 🔁 Logical Flow (Functional Blocks)

1. Data Ingestion
   - `src/data_preprocessing.py::load_csv` and `load_and_preprocess_df` read CSVs and clean text.
   - `utils/file_utils.py::extract_text` extracts text from uploaded files (PDF/DOCX/TXT) for the app.

2. Preprocessing
   - `preprocess_text` (data_preprocessing) normalizes text (remove URLs, HTML, normalize whitespace, lowercase optional).
   - `train.py::preprocess_function` tokenizes articles and summaries into model-friendly inputs and labels.

3. Training / Fine-tuning
   - `src/train.py` uses `transformers.Seq2SeqTrainer` with `AutoModelForSeq2SeqLM` and `DataCollatorForSeq2Seq`.
   - Logs training progress; computes ROUGE using `evaluate` or `rouge_score` fallback; saves model and training metadata to `outputs/model/best_model`.

4. Inference / Summarization
   - `src/model.py::load_model_and_tokenizer` and `summarize_text` wrap model loading and single-text inference using `.generate()`.
   - `src/summarize.py` and `src/generate_predictions.py` support summarizing files, folders, or CSV rows and saving outputs to `outputs/results/`.
   - `app.py` (Gradio) provides a local web UI that uses `transformers.pipeline("summarization")` for convenience and handles chunking long inputs.

5. Evaluation
   - `src/evaluate.py` computes mean ROUGE-1/2/L using `rouge_score.RougeScorer` and saves JSON results in `outputs/results/`.

6. Utilities & Logging
   - `src/config.py` centralizes paths, hyperparameters, and defaults.
   - `src/train.py::append_training_log` writes simple `training_log.csv` for later inspection.
   - `utils/file_utils.py` extracts text from common document formats.

---

## 🧭 Learning Path (If you want to recreate this)

Study these concepts and docs in order:
1. NLP basics: tokenization, seq2seq vs extractive summarization
2. Transformers & Hugging Face:
   - transformers docs (model hub, pipelines, generation API)
   - Trainer API and `Seq2SeqTrainer` patterns
3. PyTorch fundamentals (tensors, device management, training loops)
4. Datasets and evaluation:
   - Hugging Face `datasets` (data formatting, mapping, batching)
   - ROUGE metrics and how to interpret them (`rouge-score`, `evaluate`)
5. Practical engineering:
   - text preprocessing best practices (cleaning, truncation)
   - model checkpointing & reproducibility
   - building a simple inference UI (Gradio)

Recommended resources:
- Hugging Face Transformers course and docs
- Papers: BART (Lewis et al., 2019), T5 (Raffel et al.), Pegasus (Zhang et al.)
- ROUGE metric documentation and comparisons

---

## 🧩 Annotation: Complex / Non-obvious Functions

- `src/model.py::summarize_text`
  - Wraps tokenization and `model.generate()` with recommended generation arguments (beam search, length penalty, no_repeat_ngram_size).
  - Important: truncates inputs to `TRAINING['max_input_length']` to avoid OOM for long articles.

- `src/train.py::preprocess_function`
  - Tokenizes articles & targets (uses `tokenizer.as_target_tokenizer()` to get label tokenization semantics for seq2seq models; this matters for some tokenizers and training setups).

- `src/train.py::compute_rouge_batch`
  - Tries HF `evaluate` backend first; falls back to `rouge_score.RougeScorer` if needed — robust to missing dependency.

- `app.py::_prepare_input_for_pipeline` and chunking strategy
  - Naive fixed-length character chunking used for very long inputs; for production you'd want a smarter split (e.g., sentence boundaries, overlap) to preserve context across chunks.

- `utils/file_utils.py::extract_text`
  - Attempts multiple extract strategies (PDF -> DOCX -> TXT) and uses fallbacks to handle a range of uploads gracefully.

---

## ✅ Quick Notes / Improvement Ideas

- Chunking could be improved: split on sentences and add overlap so that context is preserved across chunk boundaries.
- Consider streaming generation / summarization for very long docs or hierarchical summarization.
- Add unit tests, and CI, and a clear `README.md` with usage examples.
- Provide a small example dataset (or scripts to download) for reproducible demos.

---

## Files of Interest (entry points)
- `src/train.py` — train/fine-tune model
- `src/summarize.py` — CLI summarization for files/folders
- `src/generate_predictions.py` — batch inference for CSV datasets
- `src/evaluate.py` — compute ROUGE against references
- `app.py` — Gradio web app for local inference
- `src/model.py` — model loading, summarization helper, saving
- `src/data_preprocessing.py` — ingestion and cleaning utilities


---

*Document generated from repository files on Dec 20, 2025.*
