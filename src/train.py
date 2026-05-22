# src/train.py
"""
Fine-Tuning of T5 model on CSV dataset.

Usage:
python -m src.train \
 --train_file data/train.csv \
 --val_file data/validation.csv \
 --model_name t5-small \
 --output_dir outputs/model/best_model \
 --num_train_epochs 1 \
 --per_device_train_batch_size 2 \
 --max_samples 200
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import Dataset

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AdamW,
)

from src.config import PRETRAINED_MODEL_NAME, MODEL_DIR, TRAINING, DATA, DEVICE


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# Dataset Loader
# ---------------------------
def load_csv_as_dataset(
    path: Path,
    text_col: str,
    summary_col: str,
    nrows: Optional[int] = None,
) -> Dataset:

    df = pd.read_csv(path, nrows=nrows)

    if text_col not in df.columns or summary_col not in df.columns:
        raise KeyError(
            f"Expecting columns '{text_col}' and '{summary_col}' in {path}"
        )

    df = df[[text_col, summary_col]].rename(
        columns={text_col: "article", summary_col: "summary"}
    )

    return Dataset.from_pandas(df.reset_index(drop=True))


# ---------------------------
# Tokenization
# ---------------------------
def preprocess_function(examples, tokenizer, max_input_length, max_target_length):

    inputs = examples["article"]
    targets = examples["summary"]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# ---------------------------
# Training Function
# ---------------------------
def train_loop(model, train_loader, optimizer, epoch):

    model.train()

    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):

        input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
        attention_mask = torch.tensor(batch["attention_mask"]).to(DEVICE)
        labels = torch.tensor(batch["labels"]).to(DEVICE)

        # -------------------
        # Forward Pass
        # -------------------
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # -------------------
        # Backpropagation
        # -------------------
        optimizer.zero_grad()

        loss.backward()

        # -------------------
        # Update Weights
        # -------------------
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    logger.info(f"Epoch {epoch} Training Loss: {avg_loss:.4f}")


# ---------------------------
# Validation
# ---------------------------
def eval_loop(model, val_loader, epoch):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):

            input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
            attention_mask = torch.tensor(batch["attention_mask"]).to(DEVICE)
            labels = torch.tensor(batch["labels"]).to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(val_loader)

    logger.info(f"Epoch {epoch} Validation Loss: {avg_loss:.4f}")


# ---------------------------
# Main Function
# ---------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)

    parser.add_argument("--model_name", type=str, default=PRETRAINED_MODEL_NAME)

    parser.add_argument("--output_dir", type=str, default=str(MODEL_DIR))

    parser.add_argument("--num_train_epochs", type=int, default=3)

    parser.add_argument("--per_device_train_batch_size", type=int, default=4)

    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    train_path = Path(args.train_file)
    val_path = Path(args.val_file)
    out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer and model: %s", args.model_name)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    model.to(DEVICE)

    logger.info("Loading dataset")

    train_ds = load_csv_as_dataset(
        train_path,
        DATA.get("text_column", "article"),
        DATA.get("summary_column", "highlights"),
        args.max_samples,
    )

    val_ds = load_csv_as_dataset(
        val_path,
        DATA.get("text_column", "article"),
        DATA.get("summary_column", "highlights"),
        args.max_samples,
    )

    logger.info("Tokenizing dataset")

    tokenized_train = train_ds.map(
        lambda x: preprocess_function(
            x,
            tokenizer,
            TRAINING["max_input_length"],
            TRAINING["max_target_length"],
        ),
        batched=True,
        remove_columns=train_ds.column_names,
    )

    tokenized_val = val_ds.map(
        lambda x: preprocess_function(
            x,
            tokenizer,
            TRAINING["max_input_length"],
            TRAINING["max_target_length"],
        ),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    train_loader = DataLoader(
        tokenized_train,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        tokenized_val,
        batch_size=args.per_device_train_batch_size,
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    logger.info("Starting training")

    for epoch in range(1, args.num_train_epochs + 1):

        train_loop(model, train_loader, optimizer, epoch)

        eval_loop(model, val_loader, epoch)

    logger.info("Saving model")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
    

# Model: T5ForConditionalGeneration
# Tokenizer: T5Tokenizer
# Base model: t5-small
# Framework: HuggingFace Transformers + PyTorch
# Dataset: CNN/DailyMail
# Task: Abstractive Document Summarization
