# app.py
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pathlib import Path
from datetime import datetime
import os

# change this if you want to load a different model (e.g. "t5-small")
MODEL_DIR = Path("outputs/model/best_model")

# helper to extract text from uploaded files
# expects a function `extract_text` in utils/file_utils.py
try:
    from utils.file_utils import extract_text
except Exception:
    # older path if you created utils at project root as `utils/file_utils.py`
    from file_utils import extract_text  # fallback (if you placed file_utils.py directly)
    
# create results folder if not exists
OUT_RESULTS = Path("outputs/results")
OUT_RESULTS.mkdir(parents=True, exist_ok=True)

# Load model & tokenizer (once)
device_index = 0 if torch.cuda.is_available() else -1
print("Loading model from:", MODEL_DIR)
# If MODEL_DIR is a string name ("t5-small") it will download from HF
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
# create a pipeline for convenience (pipeline handles generation parameters)
summarizer_pipeline = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=(0 if torch.cuda.is_available() else -1)
)

def _save_summary_to_file(text: str) -> str:
    """Save summary text to a timestamped file and return the path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = OUT_RESULTS / f"summary_{ts}.txt"
    fname.write_text(text, encoding="utf-8")
    return str(fname)

def _prepare_input_for_pipeline(extracted_text: str, max_chunk_len: int = 4000):
    """Split long text into chunks for safe generation; naive length-based split."""
    text = extracted_text.strip()
    if not text:
        return []
    if len(text) <= max_chunk_len:
        return [text]
    # split into chunks on roughly max_chunk_len boundaries
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_len, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks

def generate_summary_from_text(text: str, num_beams=4, min_length=30, max_length=150, length_penalty=1.0):
    """Generate summary for possibly long text (chunks -> summarize -> combine)."""
    if not text or not text.strip():
        return "No text to summarize."
    chunks = _prepare_input_for_pipeline(text, max_chunk_len=4000)
    summaries = []
    for ch in chunks:
        # call pipeline with explicit generation kwargs
        out = summarizer_pipeline(
            ch,
            num_beams=int(num_beams),
            min_length=int(min_length),
            max_length=int(max_length),
            length_penalty=float(length_penalty),
            truncation=True
        )
        if isinstance(out, list) and len(out) > 0 and "summary_text" in out[0]:
            summaries.append(out[0]["summary_text"].strip())
        else:
            summaries.append(str(out))
    # If multiple chunks, join them and optionally produce a final summary (here we join)
    final = "\n\n".join(summaries)
    return final

# ---------------------------
# Gradio functions (for UI)
# ---------------------------
def summarize_file_and_provide_download(uploaded_file, num_beams, min_length, max_length, length_penalty):
    # uploaded_file: a temporary file object from Gradio
    # Extract text
    try:
        extracted = extract_text(uploaded_file)
    except Exception as e:
        return "Failed to extract text from the uploaded file: " + str(e), None

    if not extracted or not extracted.strip():
        return "Could not extract text from the uploaded file. Try another file or paste the text directly.", None

    # Generate summary
    summary = generate_summary_from_text(extracted, num_beams, min_length, max_length, length_penalty)

    # Save summary to file for download
    try:
        saved_path = _save_summary_to_file(summary)
    except Exception as e:
        saved_path = None
        print("Warning: could not save summary file:", e)

    # Return (display_text, downloadable_file_path_or_None)
    return summary, saved_path

def summarize_pasted_text_and_provide_download(pasted_text, num_beams, min_length, max_length, length_penalty):
    if not pasted_text or not pasted_text.strip():
        return "Please paste or type some text to summarize.", None
    summary = generate_summary_from_text(pasted_text, num_beams, min_length, max_length, length_penalty)
    try:
        saved_path = _save_summary_to_file(summary)
    except Exception as e:
        saved_path = None
        print("Warning: could not save summary file:", e)
    return summary, saved_path

# ---------------------------
# Build Gradio UI
# ---------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align:center'>🧠 AI Document Summarizer</h1>")
    gr.Markdown("Upload PDF / DOCX / TXT or paste text. Adjust generation settings and download the generated summary.")

    with gr.Tabs():
        with gr.TabItem("Upload file"):
            file_in = gr.File(label="Upload file (pdf, docx, txt)")
            with gr.Row():
                nb = gr.Slider(1, 8, value=4, step=1, label="num_beams",
                               info="Beam search width. Higher = better quality but slower.")
                mn = gr.Slider(10, 200, value=30, step=5, label="min_length",
                               info="Minimum length of the generated summary (in tokens).")
                mx = gr.Slider(20, 512, value=150, step=10, label="max_length",
                               info="Maximum length of the generated summary (in tokens).")
                lp = gr.Slider(0.25, 2.0, value=1.0, step=0.25, label="length_penalty",
                               info=">1 favors shorter outputs; <1 favors longer outputs.")
            out_text = gr.Textbox(lines=8, label="Summary")
            download_file = gr.File(label="Download summary file", interactive=False)
            btn = gr.Button("Summarize file")
            btn.click(fn=summarize_file_and_provide_download,
                      inputs=[file_in, nb, mn, mx, lp],
                      outputs=[out_text, download_file])

        with gr.TabItem("Paste text"):
            pasted = gr.Textbox(lines=12, label="Paste article or text here...")
            with gr.Row():
                nb2 = gr.Slider(1, 8, value=4, step=1, label="num_beams",
                                info="Beam search width. Higher = better quality but slower.")
                mn2 = gr.Slider(10, 200, value=30, step=5, label="min_length",
                                info="Minimum length of the generated summary (in tokens).")
                mx2 = gr.Slider(20, 512, value=150, step=10, label="max_length",
                                info="Maximum length of the generated summary (in tokens).")
                lp2 = gr.Slider(0.25, 2.0, value=1.0, step=0.25, label="length_penalty",
                                info=">1 favors shorter outputs; <1 favors longer outputs.")
            out_text2 = gr.Textbox(lines=8, label="Summary")
            download_file2 = gr.File(label="Download summary file", interactive=False)
            btn2 = gr.Button("Summarize text")
            btn2.click(fn=summarize_pasted_text_and_provide_download,
                       inputs=[pasted, nb2, mn2, mx2, lp2],
                       outputs=[out_text2, download_file2])

    gr.Markdown("Model loaded from: `outputs/model/best_model`. To change model, edit `MODEL_DIR` in `app.py`.")

if __name__ == "__main__":
    # bind to localhost so http://127.0.0.1:7860 works
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
