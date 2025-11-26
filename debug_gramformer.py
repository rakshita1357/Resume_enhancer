# debug_gramformer.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_DIR = "gramformer_lora"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

print("Loaded tokenizer vocab size:", len(tokenizer))
print("Model parameters:", sum(p.numel() for p in model.parameters()))
# quick generate
text = "enhance: I worked on backend stuff"
inputs = tokenizer(text, return_tensors="pt")
out = model.generate(**inputs, max_length=100, num_beams=4)
print("Decoded:", tokenizer.decode(out[0], skip_special_tokens=True))
