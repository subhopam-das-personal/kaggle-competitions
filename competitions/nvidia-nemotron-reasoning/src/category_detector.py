"""
Category detection for Nemotron Reasoning Challenge problems.

Six categories:
  1. gravitational_constant  - physics d = 0.5*g*t^2
  2. number_base_conversion  - numeral system / Roman numerals / base-N
  3. unit_conversion         - linear ratio conversion
  4. text_encryption         - substitution cipher
  5. bit_manipulation        - bitwise operations on 8-bit integers
  6. equation_transformation - symbol-to-digit/operator bijection
"""

import re


def detect_category(prompt: str) -> str:
    """Classify a problem prompt into one of 6 categories.
    
    Returns one of:
        'gravitational_constant', 'number_base_conversion', 'unit_conversion',
        'text_encryption', 'bit_manipulation', 'equation_transformation'
    """
    p = prompt.lower()

    # Bit manipulation: look for 8-bit binary strings and bitwise keywords
    if re.search(r'[01]{8}\s*->', p) or 'bit manipulation' in p:
        return 'bit_manipulation'

    # Gravitational constant: physics / falling / gravitational
    if any(kw in p for kw in ['gravitational', 'falling distance', 'free fall',
                                'd = 0.5', 'g*t', 'gravity']):
        return 'gravitational_constant'

    # Text encryption: encrypt/decrypt + letter patterns
    if any(kw in p for kw in ['encrypt', 'decrypt', 'encryption rules',
                                'cipher', 'secret code']):
        return 'text_encryption'

    # Unit conversion: measurement / conversion / "becomes" with numbers
    if any(kw in p for kw in ['unit conversion', 'conversion factor',
                                'measurement', 'convert the following']):
        return 'unit_conversion'
    if re.search(r'[\d.]+\s*m\s+becomes\s+[\d.]+', p):
        return 'unit_conversion'
    # Also catch patterns like "X [unit] = Y" with ratio patterns
    if 'becomes' in p and re.search(r'[\d.]+.*?becomes.*?[\d.]+', p):
        return 'unit_conversion'

    # Number base / numeral system: Roman numerals, base conversion
    if any(kw in p for kw in ['numeral system', 'numeral', 'roman',
                                'base conversion', 'number system',
                                'write the number', 'secret number']):
        return 'number_base_conversion'

    # Equation transformation: transformation rules, symbol equations
    if any(kw in p for kw in ['transformation rules', 'transformation rule',
                                'determine the result', 'equation']):
        return 'equation_transformation'
    # Fallback: look for symbolic equation patterns like "α + β = γ"
    if re.search(r'[^\w\s]{1,3}\s*[+\-*/]\s*[^\w\s]{1,3}\s*=', p):
        return 'equation_transformation'

    return 'unknown'


def get_category_stats(prompts: list[str]) -> dict[str, int]:
    """Count problems per category."""
    stats = {}
    for prompt in prompts:
        cat = detect_category(prompt)
        stats[cat] = stats.get(cat, 0) + 1
    return dict(sorted(stats.items(), key=lambda x: -x[1]))
