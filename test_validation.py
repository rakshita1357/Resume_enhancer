#!/usr/bin/env python3
"""
Test script to demonstrate the enhancement validation system
"""

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

    # Check 3: Detect repetitive patterns
    words = enhanced_clean.split()
    if len(words) > 6:
        for i in range(len(words) - 7):
            phrase = ' '.join(words[i:i+4])
            rest = ' '.join(words[i+4:])
            if phrase in rest and len(phrase) > 15:
                print(f"❌ REJECTED: Contains repetition - '{phrase}' appears multiple times")
                return False

    # Check 4: Detect comma-separated list repetitions
    if ',' in enhanced:
        items = [item.strip().lower() for item in enhanced.split(',')]
        items = [item.replace(' and ', '').strip() for item in items]
        unique_items = set(items)
        if len(items) != len(unique_items):
            duplicates = [item for item in items if items.count(item) > 1]
            if duplicates:
                print(f"❌ REJECTED: Duplicate items in list: {duplicates[0]}")
                return False

    # Check 4b: Detect repeated technical terms (words before parentheses)
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

    # Check 5: Length validation
    if len(enhanced.strip()) < 10:
        print(f"❌ REJECTED: Enhanced text too short")
        return False

    if len(enhanced) > len(original) * 6:
        print(f"❌ REJECTED: Enhanced text suspiciously long")
        return False

    print(f"✅ ACCEPTED: Valid enhancement")
    return True


# Test cases based on your actual problematic outputs
test_cases = [
    {
        "original": "- Backend: Python (FastAPI, Flask), Node.js (Express)",
        "enhanced": "- Backend: Python (FastAPI, Flask), Node.js (Express) and Python (FastAPI).",
        "expected": False,
        "reason": "Should reject - contains repetition of 'Python (FastAPI)'"
    },
    {
        "original": "- Frontend: React.js, JavaScript, HTML, CSS",
        "enhanced": "- Frontend: React.js, JavaScript, CSS, HTML, CSS, HTML, CSS, and HTML.",
        "expected": False,
        "reason": "Should reject - duplicate items in list (CSS, HTML repeated)"
    },
    {
        "original": "- Databases: PostgreSQL, MongoDB, Redis",
        "enhanced": "- Databases: PostgreSQL, MongoDB, Redis, PostgreSQL, MongoDB, Redis.",
        "expected": False,
        "reason": "Should reject - exact duplicate items"
    },
    {
        "original": "Built backend APIs",
        "enhanced": "Built backend APIs",
        "expected": False,
        "reason": "Should reject - no change"
    },
    {
        "original": "I worked on data analysis",
        "enhanced": "Conducted comprehensive data analysis using Pandas and SQL, uncovering insights that improved decision-making by 30%.",
        "expected": True,
        "reason": "Should accept - genuine improvement"
    },
    {
        "original": "I built a website",
        "enhanced": "Developed a responsive e-commerce website using React and Node.js",
        "expected": True,
        "reason": "Should accept - good enhancement"
    }
]

print("="*80)
print("TESTING ENHANCEMENT VALIDATION SYSTEM")
print("="*80)

passed = 0
failed = 0

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"TEST CASE {i}: {test['reason']}")
    print(f"{'='*80}")
    print(f"ORIGINAL: {test['original']}")
    print(f"ENHANCED: {test['enhanced']}")
    print()

    result = is_valid_enhancement(test['original'], test['enhanced'])

    if result == test['expected']:
        print(f"✅ TEST PASSED")
        passed += 1
    else:
        print(f"❌ TEST FAILED - Expected {test['expected']}, got {result}")
        failed += 1
    print()

print("="*80)
print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print("="*80)

