import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from resume_filter import is_relevant_chunk

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
            # Process each line individually from the original text
            lines = text.splitlines()

            print(f"\n🔍 DEBUG: Found {len(lines)} raw lines from PDF")

            for idx, line in enumerate(lines):
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Skip very short lines (likely headers or artifacts)
                if len(line) < 15:
                    continue

                # Apply relevance filtering
                if not is_relevant_chunk(line):
                    print(f"⏭️  Line {idx+1}: SKIPPED (filtered) - {line[:60]}...")
                    continue

                # prepare input
                inp = "enhance: " + line

                # Count tokens to see if truncation happens
                token_count = len(tokenizer.encode(inp))

                print(f"\n{'='*60}")
                print(f"INPUT TO MODEL ({token_count} tokens): {inp}")
                if token_count > 128:
                    print(f"⚠️  WARNING: Input truncated from {token_count} to 128 tokens!")
                print(f"{'='*60}\n")

                inputs = tokenizer(inp, return_tensors="pt", truncation=True, max_length=128)
                outputs = model.generate(
                    **inputs,
                    max_length=256,  # Increased to allow longer outputs
                    num_beams=4,
                    early_stopping=True
                )
                enhanced = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"OUTPUT FROM MODEL: {enhanced}\n")

                original_texts.append(line)
                enhanced_texts.append(enhanced)

print(f"Writing {OUTPUT_TXT} with {len(original_texts)} enhanced chunks...")
with open(OUTPUT_TXT, "w", encoding="utf-8") as out:
    for orig, enh in zip(original_texts, enhanced_texts):
        out.write(f"ORIGINAL: {orig}\n")
        out.write(f"ENHANCED: {enh}\n\n")

print("Done.")

