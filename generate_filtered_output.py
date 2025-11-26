from resume_filter import is_relevant_chunk

IN_FILE = "filtered_chunks.txt"
OUT_FILE = "enhanced_resume_output_filtered.txt"

with open(IN_FILE, "r", encoding="utf-8") as f:
    content = f.read()

chunks = [p.strip() for p in content.split("\n\n") if p.strip()]

with open(OUT_FILE, "w", encoding="utf-8") as out:
    for ch in chunks:
        if ch.startswith("PAGE"):
            body = ':'.join(ch.split(":")[1:]).strip()
        else:
            body = ch
        if is_relevant_chunk(body):
            out.write(f"ORIGINAL: {body}\n")
            # Placeholder: not calling the model here — mark enhanced same as original to show pipeline
            out.write(f"ENHANCED: {body}\n\n")

print(f"Wrote {OUT_FILE} with relevant chunks only.")

