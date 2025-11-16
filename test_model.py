from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "resume_enhancer_t5"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def enhance(text):
    # Add prefix so the model understands the task
    formatted = "enhance: " + text.lower().strip()

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    output = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

while True:
    raw = input("\nEnter a sentence: ")
    if raw.strip().lower() == "exit":
        break
    print("Enhanced:", enhance(raw))
