# src/config.py
# Central configuration for AI_DOCUMENT_SUMMARIZER
# Edit values here to change dataset paths, model, and training hyperparameters.

from pathlib import Path
import torch

# -------------------------
# Project paths (relative)
# -------------------------
ROOT = Path(__file__).resolve().parents[1]        # project root (AI_DOCUMENT_SUMMARIZER)
DATA_DIR = ROOT / "data"
SAMPLE_TEXTS_DIR = DATA_DIR / "sample_texts"
SAMPLE_ARTICLE = SAMPLE_TEXTS_DIR / "sample_article.txt"

OUTPUTS_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUTS_DIR / "model" / "best_model"   # where model.save_pretrained(...) will write
LOGS_DIR = OUTPUTS_DIR / "logs"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Create directories if they don't exist (safe to call at import)
for p in (SAMPLE_TEXTS_DIR, MODEL_DIR, LOGS_DIR, RESULTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# -------------------------
# Model & tokenizer
# -------------------------
# Change this to "facebook/bart-large-cnn", "t5-small", "google/pegasus-xsum", etc.
PRETRAINED_MODEL_NAME = "facebook/bart-large-cnn"
TOKENIZER_NAME = PRETRAINED_MODEL_NAME  # usually same as model

# -------------------------
# Device
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Training hyperparameters (tweak as needed)
# -------------------------
TRAINING = {
    "epochs": 3,
    "train_batch_size": 4,       # per-device batch size
    "eval_batch_size": 8,
    "learning_rate": 3e-5,
    "weight_decay": 0.0,
    "adam_epsilon": 1e-8,
    "warmup_steps": 500,
    "max_grad_norm": 1.0,
    "gradient_accumulation_steps": 1,
    "max_input_length": 1024,    # truncate/pad input articles
    "max_target_length": 128,    # max tokens in generated summary
    "save_steps": 1000,          # save checkpoint every N steps (if using)
    "logging_steps": 100,
    "seed": 42,
}

# -------------------------
# Data & evaluation settings
# -------------------------
DATA = {
    "train_file": DATA_DIR / "train.csv",
    "validation_file": DATA_DIR / "validation.csv",
    "test_file": DATA_DIR / "test.csv",
    "text_column": "article",        # column name in CSV that contains article text
    "summary_column": "highlights",  # column with reference summaries
}

# -------------------------
# Utility / misc
# -------------------------
NUM_WORKERS = 4   # for DataLoader, set 0 on Windows if issues
VERBOSE = True

def print_config():
    """Print a brief config summary."""
    print("Project root:", ROOT)
    print("Data dir:", DATA_DIR)
    print("Sample article:", SAMPLE_ARTICLE)
    print("Model dir:", MODEL_DIR)
    print("Logs dir:", LOGS_DIR)
    print("Results dir:", RESULTS_DIR)
    print("Pretrained model:", PRETRAINED_MODEL_NAME)
    print("Device:", DEVICE)
    print("Training epochs:", TRAINING["epochs"])
