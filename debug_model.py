from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model_path = "resume_enhancer_t5"

print("Loading model...")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

print("Testing tokenizer...")
print("Tokenizer vocab size:", len(tokenizer))

print("Testing model weights...")
state_dict = model.state_dict()
print("Number of parameters:", sum(p.numel() for p in model.parameters()))

test_input = "enhance: i worked on backend tasks"
inputs = tokenizer(test_input, return_tensors="pt")

print("Tokenized input IDs:", inputs["input_ids"])

with torch.no_grad():
    output = model.generate(**inputs, max_length=50)

print("Raw output IDs:", output)
print("Decoded:", tokenizer.decode(output[0], skip_special_tokens=True))
