from resume_filter import is_relevant_chunk

with open("filtered_chunks.txt", "r", encoding="utf-8") as f:
    content = f.read()

chunks = [p.strip() for p in content.split("\n\n") if p.strip()]

print(f"Total chunks in filtered_chunks.txt: {len(chunks)}\n")
for i, ch in enumerate(chunks, start=1):
    # each chunk starts with PAGE N: so remove that prefix when checking relevance
    if ch.startswith("PAGE"):
        ch_body = ':'.join(ch.split(":")[1:]).strip()
    else:
        ch_body = ch
    rel = is_relevant_chunk(ch_body)
    print(f"CHUNK {i} (relevant={rel}):\n{ch_body[:400]}\n")

