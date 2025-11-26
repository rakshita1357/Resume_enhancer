import sys
import pdfplumber
from resume_filter import filter_resume_text

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else "temp_resume.pdf"
OUT_TXT = "filtered_chunks.txt"

chunks = []

with pdfplumber.open(PDF_PATH) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        cleaned = filter_resume_text(text)
        page_chunks = [c.strip() for c in cleaned.split("\n\n") if c.strip()]
        for c in page_chunks:
            chunks.append((i, c))

with open(OUT_TXT, "w", encoding="utf-8") as f:
    for page_no, c in chunks:
        f.write(f"PAGE {page_no}: {c}\n\n")

print(f"Processed PDF: {PDF_PATH}")
print(f"Pages scanned: {len(pdf.pages)}")
print(f"Chunks found: {len(chunks)}")
print(f"Filtered chunks written to: {OUT_TXT}")

