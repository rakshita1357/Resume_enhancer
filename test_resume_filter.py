from resume_filter import filter_resume_text


def test_filter_basic():
    raw = """
    Alex Carter
    Email: alex.carter@example.com
    Phone: +91 9876501122
    Location: Bengaluru, India

    Professional Summary

    Technical developer with strong experience in building web apps.

    Technical Skills

    Backend: Python (FastAPI, Flask)
    """

    cleaned = filter_resume_text(raw)

    # The cleaned text should not contain the header or contact lines
    assert "Email" not in cleaned
    assert "Phone" not in cleaned
    assert "Professional Summary" not in cleaned
    assert "Technical Skills" not in cleaned

    # It should contain the narrative sentence
    assert "Technical developer with strong experience" in cleaned


if __name__ == "__main__":
    # Quick manual run
    test_filter_basic()
    print("resume_filter test passed")

