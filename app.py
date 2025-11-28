import pdfplumber
from fastapi import FastAPI, UploadFile, File
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from resume_filter import is_relevant_chunk

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_DIR = "gramformer_lora"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

OUTPUT_TXT = "enhanced_resume_output.txt"


def enhance_line(text: str):
    text = text.strip()
    if not text or len(text) < 15:  # Skip very short lines
        return ""

    inp = "enhance: " + text

    # Count tokens to detect truncation
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
    return enhanced


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_path = "temp_resume.pdf"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    original_texts = []
    enhanced_texts = []

    # Extract lines from PDF
    with pdfplumber.open(temp_path) as pdf:
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

                    # Now enhance this individual line
                    enhanced = enhance_line(line)
                    if enhanced:  # Only add if not empty
                        original_texts.append(line)
                        enhanced_texts.append(enhanced)

    # Write to TXT file
    with open(OUTPUT_TXT, "w", encoding="utf-8") as out:
        for orig, enh in zip(original_texts, enhanced_texts):
            out.write(f"ORIGINAL: {orig}\n")
            out.write(f"ENHANCED: {enh}\n\n")

    return FileResponse(
        OUTPUT_TXT,
        media_type="text/plain",
        filename="enhanced_resume.txt"
    )
