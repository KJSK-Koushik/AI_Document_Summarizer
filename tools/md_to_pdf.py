from fpdf import FPDF
from pathlib import Path

IN = Path("docs/architecture_overview.md")
OUT = Path("outputs/architecture_overview.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=12)

if not IN.exists():
    raise FileNotFoundError("Markdown file not found: " + str(IN))

with IN.open("r", encoding="utf-8") as f:
    for line in f:
        # sanitize unicode characters that FPDF (latin-1) can't encode
        s = line.rstrip("\n")
        s = s.replace("—", "--").replace("–", "-")
        s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        # replace common emojis and symbols and normalize bullets
        s = s.replace("📌", "")
        s = s.replace("✅", "")
        s = s.replace("🧠", "")
        s = s.replace("•", "-")
        # remove remaining non-ascii characters (avoid latin1 encode errors)
        s = ''.join(ch if ord(ch) < 128 else ' ' for ch in s)
        if s.startswith("# "):
            # H1
            pdf.set_font("Arial", 'B', 16)
            pdf.ln(2)
            pdf.cell(0, 10, s[2:], ln=True)
            pdf.ln(1)
            pdf.set_font("Arial", size=12)
        elif s.startswith("## "):
            pdf.set_font("Arial", 'B', 14)
            pdf.ln(2)
            pdf.cell(0, 8, s[3:], ln=True)
            pdf.ln(1)
            pdf.set_font("Arial", size=12)
        elif s.startswith("### "):
            pdf.set_font("Arial", 'B', 12)
            pdf.ln(1)
            pdf.cell(0, 7, s[4:], ln=True)
            pdf.set_font("Arial", size=12)
        elif s.startswith("- "):
            # bullet
            text = s[2:]
            # wrap text
            pdf.set_x(20)
            # use ASCII hyphen as bullet to avoid encoding issues
            pdf.multi_cell(0, 6, "- " + text)
        elif s.strip() == "---":
            pdf.ln(2)
            pdf.cell(0, 2, "", ln=True)
            pdf.ln(1)
        else:
            if s.strip() == "":
                pdf.ln(2)
            else:
                pdf.multi_cell(0, 6, s)

pdf.output(str(OUT))
print("Wrote PDF to:", OUT)
