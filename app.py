import pdfplumber
from fastapi import FastAPI, UploadFile, File
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi.responses import FileResponse
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

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
    if not text:
        return ""

    inp = "enhance: " + text
    inputs = tokenizer(inp, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


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
                lines = text.split("\n")
                for line in lines:
                    original_texts.append(line)
                    enhanced_texts.append(enhance_line(line))

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
