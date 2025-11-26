import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from resume_filter import filter_resume_text, is_relevant_chunk

MODEL_DIR = "gramformer_lora"
OUTPUT_TXT = "enhanced_resume_output.txt"
PDF = "temp_resume.pdf"

print("Loading tokenizer and model (this may take a moment)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

print("Model loaded. Processing PDF...")
original_texts = []
enhanced_texts = []

with pdfplumber.open(PDF) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            cleaned = filter_resume_text(text)
            chunks = [c.strip() for c in cleaned.split("\n\n") if c.strip()]
            for chunk in chunks:
                if not is_relevant_chunk(chunk):
                    continue
                # prepare input
                inp = "enhance: " + chunk.strip()
                inputs = tokenizer(inp, return_tensors="pt", truncation=True)
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
                enhanced = tokenizer.decode(outputs[0], skip_special_tokens=True)
                original_texts.append(chunk)
                enhanced_texts.append(enhanced)

print(f"Writing {OUTPUT_TXT} with {len(original_texts)} enhanced chunks...")
with open(OUTPUT_TXT, "w", encoding="utf-8") as out:
    for orig, enh in zip(original_texts, enhanced_texts):
        out.write(f"ORIGINAL: {orig}\n")
        out.write(f"ENHANCED: {enh}\n\n")

print("Done.")

