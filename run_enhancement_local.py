import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from resume_filter import is_relevant_chunk

MODEL_DIR = "gramformer_lora"
OUTPUT_TXT = "enhanced_resume_output.txt"
PDF = "temp_resume.pdf"

print("Loading tokenizer and model (this may take a moment)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)


def is_valid_enhancement(original: str, enhanced: str) -> bool:
    """
    Validates if the enhanced text is actually an improvement.
    Returns False if:
    - Enhanced is identical or too similar to original
    - Enhanced contains repetitive patterns
    - Enhanced is suspiciously short or long
    """
    original_clean = original.strip().lower()
    enhanced_clean = enhanced.strip().lower()

    # Check 1: No change or minimal change
    if original_clean == enhanced_clean:
        print(f"❌ REJECTED: No change from original")
        return False

    # Check 2: Enhanced text is just a substring or slightly modified
    if original_clean in enhanced_clean and len(enhanced_clean) < len(original_clean) * 1.2:
        print(f"❌ REJECTED: Minimal modification")
        return False

    # Check 3: Detect repetitive patterns (same phrase repeated)
    # Split into words and check for consecutive repeated segments
    words = enhanced_clean.split()
    if len(words) > 6:
        # Check for repeated 4+ word phrases (more reliable than 3-word)
        for i in range(len(words) - 7):
            phrase = ' '.join(words[i:i+4])
            rest = ' '.join(words[i+4:])
            if phrase in rest and len(phrase) > 15:  # Only check substantial phrases
                print(f"❌ REJECTED: Contains repetition - '{phrase}' appears multiple times")
                return False

    # Check 4: Detect comma-separated list repetitions (like "CSS, HTML, CSS, HTML")
    if ',' in enhanced:
        items = [item.strip().lower() for item in enhanced.split(',')]
        # Remove "and" from last item if present
        items = [item.replace(' and ', '').strip() for item in items]
        # Check for duplicates
        unique_items = set(items)
        if len(items) != len(unique_items):
            # Find the duplicates
            duplicates = [item for item in items if items.count(item) > 1]
            if duplicates:
                print(f"❌ REJECTED: Duplicate items in list: {duplicates[0]}")
                return False

    # Check 4b: Detect repeated technical terms (words before parentheses)
    # Look for patterns like "Python (...)" appearing multiple times
    import re
    words_split = enhanced_clean.split()
    words_before_paren = []
    for i, word in enumerate(words_split):
        if '(' in word or (i < len(words_split)-1 and words_split[i+1].startswith('(')):
            clean = word.replace(':', '').replace(',', '').replace('-', '').strip()
            if clean and not clean.startswith('('):
                words_before_paren.append(clean)
    if words_before_paren and len(words_before_paren) != len(set(words_before_paren)):
        print(f"❌ REJECTED: Repeated technical term before parentheses")
        return False

    # Check 5: Too short (less than 10 chars) or suspiciously long (>6x original)
    if len(enhanced.strip()) < 10:
        print(f"❌ REJECTED: Enhanced text too short")
        return False

    # Be more lenient with length - training data shows good enhancements can be 5-6x longer
    if len(enhanced) > len(original) * 6:
        print(f"❌ REJECTED: Enhanced text suspiciously long ({len(enhanced)} vs {len(original)} chars)")
        return False

    # All checks passed
    print(f"✅ ACCEPTED: Valid enhancement")
    return True


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

                # Validate the enhancement
                if is_valid_enhancement(line, enhanced):
                    original_texts.append(line)
                    enhanced_texts.append(enhanced)
                else:
                    # Enhancement was rejected, keep original
                    print(f"⚠️  Keeping original text instead\n")
                    original_texts.append(line)
                    enhanced_texts.append(line)  # Use original as fallback

print(f"Writing {OUTPUT_TXT} with {len(original_texts)} enhanced chunks...")
with open(OUTPUT_TXT, "w", encoding="utf-8") as out:
    for orig, enh in zip(original_texts, enhanced_texts):
        out.write(f"ORIGINAL: {orig}\n")
        out.write(f"ENHANCED: {enh}\n\n")

print("\n" + "="*60)
print("ENHANCEMENT STATISTICS")
print("="*60)
print(f"Total lines processed: {stats['processed']}")
print(f"Valid enhancements:    {stats['accepted']} ({stats['accepted']/stats['processed']*100:.1f}%)")
print(f"Rejected (kept orig):  {stats['rejected']} ({stats['rejected']/stats['processed']*100:.1f}%)")
print("="*60)
print("Done.")

