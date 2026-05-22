# src/model.py
"""
Model and tokenizer utilities for AI_DOCUMENT_SUMMARIZER.

Provides:
- load_model_and_tokenizer(pretrained_name=None, device=None)
- summarize_text(text, model, tokenizer, max_length=None, min_length=None, num_beams=4)
- save_model(model, tokenizer, out_dir)

Example:
    from src.config import PRETRAINED_MODEL_NAME, DEVICE, SAMPLE_ARTICLE
    model, tokenizer = load_model_and_tokenizer()
    article = SAMPLE_ARTICLE.read_text(encoding="utf-8")
    summary = summarize_text(article, model, tokenizer)
    print(summary)
"""
from pathlib import Path
from typing import Optional, Tuple, List, Union

# Explicit T5 imports 
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
except Exception as e:
    raise ImportError(
        "transformers is required. Install it with `pip install transformers`.\n"
        "Original error: " + str(e)
    )

try:
    import torch
except Exception as e:
    raise ImportError(
        "PyTorch is required. Install it with `pip install torch`.\n"
        "Original error: " + str(e)
    )

# Optional: default settings if config not available
try:
    from src.config import PRETRAINED_MODEL_NAME, DEVICE, TRAINING, MODEL_DIR
except Exception:
    PRETRAINED_MODEL_NAME = "t5-small"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAINING = {"max_input_length": 1024, "max_target_length": 128}
    MODEL_DIR = Path("outputs/model/best_model")


def load_model_and_tokenizer(
    pretrained_name: Optional[str] = None,
    device: Optional[str] = None,
    local_dir: Optional[Union[str, Path]] = None,
) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """
    Load a T5 model and tokenizer.

    Args:
        pretrained_name: Hugging Face model id (default: t5-small)
        device: "cpu" or "cuda"
        local_dir: load model from a local directory if provided

    Returns:
        model (moved to device), tokenizer
    """

    pretrained = pretrained_name or PRETRAINED_MODEL_NAME
    device = device or DEVICE

    if local_dir:
        model_path = str(local_dir)

        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)

    else:
        tokenizer = T5Tokenizer.from_pretrained(pretrained)
        model = T5ForConditionalGeneration.from_pretrained(pretrained)

    model.to(device)
    model.eval()

    return model, tokenizer


def _chunk_and_tokenize(
    texts: List[str],
    tokenizer,
    max_length: int,
):
    """
    Tokenize text safely with truncation.
    """

    return tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=max_length,
    )


def summarize_text(
    text: str,
    model,
    tokenizer,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    num_beams: int = 4,
    length_penalty: float = 2.0,
    early_stopping: bool = True,
    device: Optional[str] = None,
) -> str:
    """
    Generate a summary for a single text string.
    """

    device = device or DEVICE

    max_input_len = TRAINING.get("max_input_length", 1024)
    max_tgt_len = max_length or TRAINING.get("max_target_length", 128)

    inputs = _chunk_and_tokenize([text], tokenizer, max_length=max_input_len)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
#generation paramters
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_tgt_len,
        num_beams=num_beams,
        length_penalty=length_penalty,#Controls summary length preference
        early_stopping=early_stopping,
        no_repeat_ngram_size=3,
        do_sample=False,
    )

    if min_length is not None:
        gen_kwargs["min_length"] = min_length

    with torch.no_grad():
        generated = model.generate(**gen_kwargs)

    summary = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return summary[0].strip()


def save_model(model, tokenizer, out_dir: Union[str, Path]):
    """
    Save model and tokenizer to directory.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


if __name__ == "__main__":

    try:
        from src.config import SAMPLE_ARTICLE
        sample_path = SAMPLE_ARTICLE
    except Exception:
        sample_path = Path("data/sample_texts/sample_article.txt")

    print("Loading model and tokenizer...")

    model, tokenizer = load_model_and_tokenizer()

    if not Path(sample_path).exists():
        raise FileNotFoundError(f"Sample article not found at {sample_path}")

    raw_text = Path(sample_path).read_text(encoding="utf-8")

    print("\n--- RAW INPUT (first 800 chars) ---\n")
    print(raw_text[:800])

    print("\nGenerating summary...\n")

    summary = summarize_text(raw_text, model, tokenizer)

    print("\n--- GENERATED SUMMARY ---\n")
    print(summary)
    
    
    
# Architecture : T5
# Tokenizer    : T5Tokenizer
# Model        : T5ForConditionalGeneration
# Base Model   : t5-small
# Framework    : HuggingFace Transformers + PyTorch
# Task         : Abstractive Document Summarization