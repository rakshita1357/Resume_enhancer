import re
from typing import List


def filter_resume_text(extracted_text: str) -> str:
    """
    Cleans extracted resume text by removing structural elements (headers and contact info).

    Args:
        extracted_text: The raw text extracted from the resume.

    Returns:
        A string containing only the narrative content suitable for enhancement.
    """

    if not extracted_text:
        return ""

    # --- 1. Regex Patterns ---

    # Pattern for common resume headers (stand-alone lines).
    HEADER_PATTERN = re.compile(
        r"^\s*(Professional Summary|Technical Skills|Skills|Work Experience|Experience|Major Projects|Projects|Education|Achievements|Hobbies|Career Objective|Summary|About|Certifications)\s*[:\-]?\s*$",
        re.IGNORECASE,
    )

    # Pattern for contact information lines.
    CONTACT_PATTERN = re.compile(
        r"^\s*(Email|E-mail|Phone|Mobile|Contact|Location|Address|LinkedIn|GitHub|Website|Portfolio)[\s:-]*.*",
        re.IGNORECASE,
    )

    # Also filter out common single-line artifacts like separators or page numbers
    ARTIFACT_PATTERN = re.compile(r"^[-=_]{2,}$|^Page\s+\d+$", re.IGNORECASE)

    # --- 2. Filtering Logic ---

    lines = extracted_text.splitlines()
    filtered_lines: List[str] = []

    for line in lines:
        # Strip leading/trailing whitespace for clean matching
        cleaned_line = line.strip()

        # Skip empty lines
        if not cleaned_line:
            continue

        # Check for contact info
        if CONTACT_PATTERN.search(cleaned_line):
            continue

        # Check for headers (stand-alone section labels)
        if HEADER_PATTERN.match(cleaned_line):
            continue

        # Skip visual separators / page numbers
        if ARTIFACT_PATTERN.match(cleaned_line):
            continue

        # If the line is not a header/contact/artifact, keep it
        filtered_lines.append(cleaned_line)

    # Join lines but keep paragraph separation: detect paragraphs by original empty lines
    # Since we removed empty lines above, reconstruct paragraphs by grouping contiguous lines
    paragraphs: List[str] = []
    current_para: List[str] = []

    for ln in filtered_lines:
        if ln == "":
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
        else:
            current_para.append(ln)

    if current_para:
        paragraphs.append(" ".join(current_para))

    # Return paragraphs separated by double newlines so callers can split into chunks
    return '\n\n'.join(paragraphs)


# New helper to allow an extra safety check before sending text to the model
def is_relevant_chunk(chunk: str, min_words: int = 3) -> bool:
    """
    Heuristic to decide if a chunk should be sent to the enhancement model.
    Returns False for chunks that look like contact lines, headers, separators, are too short, or look like a name-only line.
    """
    if not chunk or not chunk.strip():
        return False

    text = chunk.strip()

    # Quick length check
    words = text.split()
    if len(words) < min_words:
        return False

    # Patterns reused from the main filter
    header_re = re.compile(r"^\s*(Professional Summary|Technical Skills|Skills|Work Experience|Experience|Major Projects|Projects|Education|Achievements|Hobbies|Career Objective|Summary|About|Certifications)\b", re.IGNORECASE)
    contact_re = re.compile(r"^\s*(Email|E-mail|Phone|Mobile|Contact|Location|Address|LinkedIn|GitHub|Website|Portfolio)\b", re.IGNORECASE)
    artifact_re = re.compile(r"^[-=_]{2,}$|^Page\s+\d+$", re.IGNORECASE)

    if contact_re.search(text):
        return False
    if header_re.search(text):
        return False
    if artifact_re.search(text):
        return False

    # Drop bullet-only lines or lines that are just punctuation
    if re.match(r"^[\-\u2022\*\s]+$", text):
        return False

    # Heuristic: drop name-only lines: short (<=3 words) and Title Cased words
    if len(words) <= 3:
        titlecased_words = sum(1 for w in words if re.match(r"^[A-Z][a-z]+$", w))
        if titlecased_words == len(words):
            # looks like a person's name or short title like "John Doe" or "Alex Carter"
            return False

    # If it passes all checks, consider it relevant
    return True


__all__ = ["filter_resume_text", "is_relevant_chunk"]
