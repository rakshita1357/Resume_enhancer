# test_gramformer.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_DIR = "gramformer_lora"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

def enhance(text):
    inp = "enhance: " + text.strip()
    inputs = tokenizer(inp, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        s = input("Enter sentence (or exit): ")
        if s.strip().lower() == "exit":
            break
        print("Enhanced:", enhance(s))
