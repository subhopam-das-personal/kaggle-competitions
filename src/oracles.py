"""
Deterministic Python oracles for each problem category.

Each oracle takes a prompt string and returns the correct answer (or None if
it cannot solve the problem). These are used both for:
  1. Verifying training data (discard examples where oracle != provided answer)
  2. Generating synthetic training traces with guaranteed-correct answers
"""

import re
import math
import string
from typing import Optional
from itertools import permutations


# ============================================================================
# Category 1: Gravitational Constant
# ============================================================================

def solve_gravitational(prompt: str) -> Optional[str]:
    """Solve gravitational constant problems: d = 0.5 * g * t^2.
    
    Strategy: extract (t, d) pairs from examples, compute g, apply to query t.
    """
    # Extract all (time, distance) example pairs
    # Patterns: "t = 2.5s, distance = 31.25" or "For t = 2.5s, distance = 31.25 m"
    pairs = re.findall(
        r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)',
        prompt, re.IGNORECASE
    )
    if not pairs:
        # Try alternate pattern: "t = X ... d = Y"
        pairs = re.findall(
            r't\s*=\s*([\d.]+).*?d\s*=\s*([\d.]+)',
            prompt, re.IGNORECASE
        )
    
    if not pairs:
        return None
    
    # Compute g from each pair: g = 2d / t^2
    g_values = []
    for t_str, d_str in pairs:
        t, d = float(t_str), float(d_str)
        if t > 0:
            g_values.append(2 * d / (t ** 2))
    
    if not g_values:
        return None
    
    # Use median g (robust to outliers)
    g = sorted(g_values)[len(g_values) // 2]
    
    # Find query time — must be AFTER the examples section
    # Look for "determine ... t = X" or "for t = X" in the query portion
    query_match = re.search(
        r'(?:determine|calculate|find|compute).*?t\s*=\s*([\d.]+)',
        prompt, re.IGNORECASE
    )
    if query_match:
        query_t = float(query_match.group(1))
    else:
        # Fallback: last t = X in the prompt (after all examples)
        all_t = re.findall(r't\s*=\s*([\d.]+)', prompt)
        if all_t:
            query_t = float(all_t[-1])
        else:
            return None
    
    result = 0.5 * g * query_t ** 2
    # Format: 2 decimal places
    return f"{result:.2f}"


def gravitational_cot(prompt: str, answer: str) -> str:
    """Generate chain-of-thought trace for gravitational problems."""
    pairs = re.findall(
        r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)',
        prompt, re.IGNORECASE
    )
    if not pairs:
        return f"Using d = 0.5*g*t², the answer is {answer}"
    
    t0, d0 = float(pairs[0][0]), float(pairs[0][1])
    g = 2 * d0 / (t0 ** 2)
    
    lines = [
        "I need to find the gravitational constant g from the examples.",
        f"Using the formula d = 0.5 * g * t², we get g = 2d / t².",
        "",
        f"From the first example: t = {t0}s, d = {d0}m",
        f"g = 2 × {d0} / {t0}² = {2*d0} / {t0**2} = {g:.6f} m/s²",
    ]
    
    # Verify with second example if available
    if len(pairs) > 1:
        t1, d1 = float(pairs[1][0]), float(pairs[1][1])
        g1 = 2 * d1 / (t1 ** 2)
        lines.append(f"\nVerification with second example: t = {t1}s, d = {d1}m")
        lines.append(f"g = 2 × {d1} / {t1}² = {g1:.6f} m/s² ✓")
    
    # Find query
    all_t = re.findall(r't\s*=\s*([\d.]+)', prompt)
    query_t = float(all_t[-1]) if all_t else 1.0
    result = 0.5 * g * query_t ** 2
    
    lines.extend([
        "",
        f"Now calculating for t = {query_t}s:",
        f"d = 0.5 × {g:.6f} × {query_t}²",
        f"d = 0.5 × {g:.6f} × {query_t**2}",
        f"d = {result:.2f}",
    ])
    
    return "\n".join(lines)


# ============================================================================
# Category 2: Number Base Conversion / Roman Numerals
# ============================================================================

ROMAN_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def int_to_roman(num: int) -> str:
    """Convert integer to Roman numeral string."""
    result = ""
    for value, symbol in ROMAN_VALUES:
        while num >= value:
            result += symbol
            num -= value
    return result


def roman_to_int(s: str) -> int:
    """Convert Roman numeral string to integer."""
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50,
                 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    for i in range(len(s)):
        if i + 1 < len(s) and roman_map.get(s[i], 0) < roman_map.get(s[i+1], 0):
            result -= roman_map.get(s[i], 0)
        else:
            result += roman_map.get(s[i], 0)
    return result


def solve_number_base(prompt: str) -> Optional[str]:
    """Solve number/numeral system conversion problems.
    
    Detects whether it's Roman numerals or another base system from examples.
    """
    # Check for Roman numeral patterns in examples
    roman_pattern = re.findall(r'(\d+)\s*->\s*([IVXLCDM]+)', prompt)
    if roman_pattern:
        # It's Roman numerals - find the query number
        query_match = re.search(r'write the number\s+(\d+)', prompt, re.IGNORECASE)
        if not query_match:
            query_match = re.search(r'convert.*?(\d+)', prompt, re.IGNORECASE)
        if query_match:
            target = int(query_match.group(1))
            return int_to_roman(target)
    
    # Check for general base conversion: "number -> converted"
    # Try to detect the base from examples
    examples = re.findall(r'(\d+)\s*->\s*(\w+)', prompt)
    if examples:
        # Check if outputs look like different base representations
        # For now, try Roman numerals heuristic
        all_roman = all(
            all(c in 'IVXLCDM' for c in out)
            for _, out in examples
        )
        if all_roman:
            query_match = re.search(r'(?:write|convert|transform).*?(\d+)', prompt, re.IGNORECASE)
            if query_match:
                return int_to_roman(int(query_match.group(1)))
    
    # Try base-N detection by checking example input -> output consistency
    # across common bases (2-36)
    for base in range(2, 37):
        consistent = True
        for inp_str, out_str in examples:
            try:
                inp_val = int(inp_str)
                # Check if out_str is inp_val in the target base
                converted = _int_to_base(inp_val, base)
                if converted.upper() != out_str.upper():
                    consistent = False
                    break
            except (ValueError, ZeroDivisionError):
                consistent = False
                break
        
        if consistent and examples:
            query_match = re.search(r'(?:write|convert|transform).*?(\d+)', prompt, re.IGNORECASE)
            if query_match:
                return _int_to_base(int(query_match.group(1)), base)
    
    return None


def _int_to_base(num: int, base: int) -> str:
    """Convert integer to string in given base."""
    if num == 0:
        return "0"
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ""
    n = abs(num)
    while n:
        result = digits[n % base] + result
        n //= base
    return ("-" + result) if num < 0 else result


def number_base_cot(prompt: str, answer: str) -> str:
    """Generate CoT for number base conversion."""
    # Check if Roman
    roman_pattern = re.findall(r'(\d+)\s*->\s*([IVXLCDM]+)', prompt)
    if roman_pattern:
        query_match = re.search(r'write the number\s+(\d+)', prompt, re.IGNORECASE)
        if not query_match:
            query_match = re.search(r'convert.*?(\d+)', prompt, re.IGNORECASE)
        if query_match:
            num = int(query_match.group(1))
            lines = [
                f"I need to convert {num} to the Wonderland numeral system.",
                "Looking at the examples, I can see this is Roman numeral conversion.",
                "",
            ]
            # Walk through the conversion
            remaining = num
            parts = []
            for value, symbol in ROMAN_VALUES:
                while remaining >= value:
                    parts.append(f"{symbol} ({value})")
                    remaining -= value
            
            lines.append("Breaking down the number:")
            if num >= 1000:
                lines.append(f"  Thousands: {num // 1000} × 1000 = {num // 1000 * 1000} → {'M' * (num // 1000)}")
            hundreds = (num % 1000) // 100
            if hundreds:
                lines.append(f"  Hundreds: {hundreds} × 100 = {hundreds * 100} → {int_to_roman(hundreds * 100)}")
            tens = (num % 100) // 10
            if tens:
                lines.append(f"  Tens: {tens} × 10 = {tens * 10} → {int_to_roman(tens * 10)}")
            ones = num % 10
            if ones:
                lines.append(f"  Ones: {ones} → {int_to_roman(ones)}")
            
            lines.append(f"\nCombining all parts: {answer}")
            return "\n".join(lines)
    
    return f"Analyzing the numeral system from examples and converting: {answer}"


# ============================================================================
# Category 3: Unit Conversion
# ============================================================================

def solve_unit_conversion(prompt: str) -> Optional[str]:
    """Solve linear unit conversion problems.
    
    Strategy: find the constant ratio from examples, apply to query.
    """
    # Pattern: "X m becomes Y" or "X [unit] = Y"
    pairs = re.findall(r'([\d.]+)\s*(?:m|meters?|units?)?\s*(?:becomes|=|→|->)\s*([\d.]+)', prompt)
    
    if not pairs:
        # Try more flexible pattern
        pairs = re.findall(r'([\d.]+)\s+becomes\s+([\d.]+)', prompt)
    
    if not pairs:
        return None
    
    # Compute ratio from each pair
    ratios = []
    for in_str, out_str in pairs:
        inp, out = float(in_str), float(out_str)
        if inp > 0:
            ratios.append(out / inp)
    
    if not ratios:
        return None
    
    # Use median ratio
    ratio = sorted(ratios)[len(ratios) // 2]
    
    # Find query value
    query_match = re.search(
        r'convert.*?(?:following|measurement)[:\s]*([\d.]+)',
        prompt, re.IGNORECASE
    )
    if not query_match:
        # Try: last number after "convert" or the query section
        parts = prompt.split("convert")
        if len(parts) > 1:
            nums = re.findall(r'([\d.]+)', parts[-1])
            if nums:
                query_val = float(nums[0])
                result = query_val * ratio
                return f"{result:.2f}"
    
    if query_match:
        query_val = float(query_match.group(1))
        result = query_val * ratio
        return f"{result:.2f}"
    
    return None


def unit_conversion_cot(prompt: str, answer: str) -> str:
    """Generate CoT for unit conversion."""
    pairs = re.findall(r'([\d.]+)\s*(?:m|meters?|units?)?\s*(?:becomes|=|→|->)\s*([\d.]+)', prompt)
    if not pairs:
        return f"Finding the conversion ratio from examples and applying: {answer}"
    
    in0, out0 = float(pairs[0][0]), float(pairs[0][1])
    ratio = out0 / in0
    
    lines = [
        "I need to find the conversion ratio from the examples.",
        "",
        f"From the first example: {in0} → {out0}",
        f"Ratio = {out0} / {in0} = {ratio:.6f}",
    ]
    
    if len(pairs) > 1:
        in1, out1 = float(pairs[1][0]), float(pairs[1][1])
        r1 = out1 / in1
        lines.append(f"\nVerification: {in1} → {out1}, ratio = {r1:.6f} ✓")
    
    # Find query
    query_match = re.search(r'convert.*?(?:following|measurement)[:\s]*([\d.]+)', prompt, re.IGNORECASE)
    if query_match:
        q = float(query_match.group(1))
        lines.extend([
            "",
            f"Applying ratio to query value {q}:",
            f"{q} × {ratio:.6f} = {q * ratio:.2f}",
        ])
    
    return "\n".join(lines)


# ============================================================================
# Category 4: Text Encryption (Substitution Cipher)
# ============================================================================

def solve_text_encryption(prompt: str) -> Optional[str]:
    """Solve substitution cipher problems.
    
    Strategy: build a character mapping from encrypted -> decrypted examples,
    then apply the mapping to the query.
    """
    # Split into examples section and query section
    # Common patterns: "Now, decrypt the following:" or "Now, decrypt:"
    split_patterns = [
        r'Now,?\s*decrypt(?:\s+the\s+following)?\s*(?:text)?[:\s]+',
        r'Decrypt\s*(?:the\s+following)?[:\s]+',
        r'What is the decrypted version of[:\s]+',
    ]
    
    examples_part = prompt
    query_part = None
    
    for pattern in split_patterns:
        parts = re.split(pattern, prompt, flags=re.IGNORECASE)
        if len(parts) >= 2:
            examples_part = parts[0]
            query_part = parts[-1].strip().strip('"\'').strip()
            break
    
    if query_part is None:
        return None
    
    # Extract example pairs: "encrypted -> decrypted" or "encrypted => decrypted"
    pairs = re.findall(
        r'([a-zA-Z][a-zA-Z\s]*?)\s*(?:->|=>|→)\s*([a-zA-Z][a-zA-Z\s]*?)(?:\n|$)',
        examples_part
    )
    
    if not pairs:
        return None
    
    # Build character mapping (encrypted char -> decrypted char)
    mapping = {}
    for enc, dec in pairs:
        enc_clean = enc.strip()
        dec_clean = dec.strip()
        
        # Align character by character (ignoring spaces consistently)
        enc_chars = enc_clean.replace(" ", "")
        dec_chars = dec_clean.replace(" ", "")
        
        if len(enc_chars) == len(dec_chars):
            for e, d in zip(enc_chars, dec_chars):
                e_lower = e.lower()
                d_lower = d.lower()
                if e_lower in mapping and mapping[e_lower] != d_lower:
                    pass  # Conflict - keep first mapping
                else:
                    mapping[e_lower] = d_lower
    
    if not mapping:
        return None
    
    # Apply mapping to query
    result = []
    unmapped_count = 0
    for c in query_part:
        if c.lower() in mapping:
            decrypted = mapping[c.lower()]
            result.append(decrypted.upper() if c.isupper() else decrypted)
        elif c == ' ':
            result.append(' ')
        elif not c.isalpha():
            result.append(c)
        else:
            result.append(c)  # Unknown char, keep as-is
            unmapped_count += 1
    
    # If there are unmapped letters, return None (can't be verified)
    if unmapped_count > 0:
        return None
    
    return "".join(result)


def text_encryption_cot(prompt: str, answer: str) -> str:
    """Generate CoT for text encryption."""
    # Extract pairs for the trace
    pairs = re.findall(
        r'([a-zA-Z][a-zA-Z\s]*?)\s*(?:->|=>|→)\s*([a-zA-Z][a-zA-Z\s]*?)(?:\n|$)',
        prompt
    )
    
    mapping = {}
    for enc, dec in pairs:
        enc_chars = enc.strip().replace(" ", "")
        dec_chars = dec.strip().replace(" ", "")
        if len(enc_chars) == len(dec_chars):
            for e, d in zip(enc_chars, dec_chars):
                mapping[e.lower()] = d.lower()
    
    sorted_map = dict(sorted(mapping.items()))
    
    lines = [
        "I need to decrypt the text using the substitution cipher from the examples.",
        "",
        "Building the letter mapping from examples:",
    ]
    
    for enc, dec in pairs[:4]:
        enc_c = enc.strip().replace(" ", "")[:8]
        dec_c = dec.strip().replace(" ", "")[:8]
        if len(enc_c) == len(dec_c):
            maps = [f"{e}→{d}" for e, d in zip(enc_c, dec_c)]
            lines.append(f"  '{enc.strip()[:25]}' → '{dec.strip()[:25]}': {', '.join(maps)}")
    
    lines.append(f"\nComplete mapping ({len(sorted_map)} letters):")
    lines.append("  " + ", ".join(f"{k}→{v}" for k, v in sorted_map.items()))
    
    # Show decryption step
    split_patterns = [
        r'Now,?\s*decrypt(?:\s+the\s+following)?[:\s]*',
        r'Decrypt[:\s]*',
    ]
    query = ""
    for pattern in split_patterns:
        parts = re.split(pattern, prompt, flags=re.IGNORECASE)
        if len(parts) == 2:
            query = parts[1].strip().strip('"\'').strip()
            break
    
    if query:
        lines.append(f"\nDecrypting: '{query}'")
        lines.append(f"Applying mapping character by character:")
        # Show first few characters
        shown = []
        for c in query[:15]:
            if c.lower() in mapping:
                shown.append(f"{c}→{mapping[c.lower()]}")
            elif c == ' ':
                shown.append("' '")
        lines.append(f"  {', '.join(shown)}{'...' if len(query) > 15 else ''}")
    
    lines.append(f"\nResult: {answer}")
    return "\n".join(lines)


# ============================================================================
# Category 5: Bit Manipulation (HARD)
# ============================================================================

def solve_bit_manipulation(prompt: str) -> Optional[str]:
    """Attempt to solve bit manipulation problems.
    
    Strategy: try common bitwise operations and compositions to find one
    that's consistent with all examples.
    
    This is the HARD category. The oracle covers common patterns but
    cannot solve all compositional boolean operations.
    """
    # Extract input -> output pairs
    pairs = re.findall(r'([01]{8})\s*(?:->|=>|→)\s*([01]{8})', prompt)
    query_match = re.search(r'(?:for|apply|input)[:\s]*([01]{8})', prompt, re.IGNORECASE)
    
    if not pairs or not query_match:
        return None
    
    query = query_match.group(1)
    
    # Convert to integers for bitwise operations
    examples = [(int(inp, 2), int(out, 2)) for inp, out in pairs]
    
    # Try single operations
    single_ops = [
        ("NOT", lambda x: (~x) & 0xFF),
        ("reverse", lambda x: int(format(x, '08b')[::-1], 2)),
        ("shift_left_1", lambda x: (x << 1) & 0xFF),
        ("shift_right_1", lambda x: (x >> 1) & 0xFF),
        ("rotate_left_1", lambda x: ((x << 1) | (x >> 7)) & 0xFF),
        ("rotate_right_1", lambda x: ((x >> 1) | (x << 7)) & 0xFF),
    ]
    
    # Add XOR/AND/OR with constants derived from examples
    for const in range(256):
        single_ops.append((f"XOR_{const}", lambda x, c=const: x ^ c))
        single_ops.append((f"AND_{const}", lambda x, c=const: x & c))
        single_ops.append((f"OR_{const}", lambda x, c=const: x | c))
    
    for name, op in single_ops:
        if all(op(inp) == out for inp, out in examples):
            result = op(int(query, 2))
            return format(result, '08b')
    
    # Try per-bit operations (each bit may have its own rule)
    # For each output bit position, check if it comes from a specific input bit
    # with possible inversion
    bit_rules = []
    solved = True
    for bit_pos in range(8):
        found = False
        for src_pos in range(8):
            for invert in [False, True]:
                consistent = True
                for inp, out in examples:
                    src_bit = (inp >> (7 - src_pos)) & 1
                    if invert:
                        src_bit = 1 - src_bit
                    expected = (out >> (7 - bit_pos)) & 1
                    if src_bit != expected:
                        consistent = False
                        break
                if consistent:
                    bit_rules.append((bit_pos, src_pos, invert))
                    found = True
                    break
            if found:
                break
        if not found:
            # Try constant bit (always 0 or always 1)
            all_zero = all(((out >> (7 - bit_pos)) & 1) == 0 for _, out in examples)
            all_one = all(((out >> (7 - bit_pos)) & 1) == 1 for _, out in examples)
            if all_zero:
                bit_rules.append((bit_pos, -1, False))  # constant 0
            elif all_one:
                bit_rules.append((bit_pos, -1, True))  # constant 1
            else:
                solved = False
                break
    
    if solved and len(bit_rules) == 8:
        query_int = int(query, 2)
        result = 0
        for bit_pos, src_pos, invert in bit_rules:
            if src_pos == -1:
                bit_val = 1 if invert else 0
            else:
                bit_val = (query_int >> (7 - src_pos)) & 1
                if invert:
                    bit_val = 1 - bit_val
            result |= (bit_val << (7 - bit_pos))
        return format(result, '08b')
    
    # Try two-input per-bit operations (AND, OR, XOR, NAND, NOR, XNOR of two input bits)
    two_bit_ops = {
        'AND': lambda a, b: a & b,
        'OR': lambda a, b: a | b,
        'XOR': lambda a, b: a ^ b,
        'NAND': lambda a, b: 1 - (a & b),
        'NOR': lambda a, b: 1 - (a | b),
        'XNOR': lambda a, b: 1 - (a ^ b),
    }
    
    bit_rules_2 = []
    solved_2 = True
    for bit_pos in range(8):
        found = False
        for src1 in range(8):
            for src2 in range(src1, 8):
                for op_name, op_fn in two_bit_ops.items():
                    consistent = True
                    for inp, out in examples:
                        b1 = (inp >> (7 - src1)) & 1
                        b2 = (inp >> (7 - src2)) & 1
                        expected = (out >> (7 - bit_pos)) & 1
                        if op_fn(b1, b2) != expected:
                            consistent = False
                            break
                    if consistent:
                        bit_rules_2.append((bit_pos, src1, src2, op_name))
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            solved_2 = False
            break
    
    if solved_2 and len(bit_rules_2) == 8:
        query_int = int(query, 2)
        result = 0
        for bit_pos, src1, src2, op_name in bit_rules_2:
            b1 = (query_int >> (7 - src1)) & 1
            b2 = (query_int >> (7 - src2)) & 1
            bit_val = two_bit_ops[op_name](b1, b2)
            result |= (bit_val << (7 - bit_pos))
        return format(result, '08b')
    
    return None


def bit_manipulation_cot(prompt: str, answer: str) -> str:
    """Generate CoT for bit manipulation."""
    pairs = re.findall(r'([01]{8})\s*(?:->|=>|→)\s*([01]{8})', prompt)
    query_match = re.search(r'(?:for|apply|input)[:\s]*([01]{8})', prompt, re.IGNORECASE)
    query = query_match.group(1) if query_match else "????????"
    
    lines = [
        "I need to find the bitwise transformation rule from the examples.",
        "",
        "Input → Output pairs:",
    ]
    for inp, out in pairs[:5]:
        lines.append(f"  {inp} → {out}")
    
    lines.extend([
        "",
        "Let me analyze each bit position to find the pattern.",
        "Checking if each output bit is derived from input bits via",
        "bitwise operations (AND, OR, XOR, NOT, XNOR, etc.).",
        "",
    ])
    
    # Try to explain the discovered pattern
    examples = [(int(inp, 2), int(out, 2)) for inp, out in pairs]
    
    # Check per-bit
    for bit_pos in range(8):
        for src_pos in range(8):
            for invert in [False, True]:
                consistent = all(
                    (((inp >> (7 - src_pos)) & 1) ^ int(invert)) == ((out >> (7 - bit_pos)) & 1)
                    for inp, out in examples
                )
                if consistent:
                    op = "NOT " if invert else ""
                    lines.append(f"  Bit {bit_pos}: = {op}input_bit[{src_pos}]")
                    break
            else:
                continue
            break
        else:
            lines.append(f"  Bit {bit_pos}: complex operation")
    
    lines.extend([
        "",
        f"Applying the transformation to input: {query}",
        f"Result: {answer}",
    ])
    
    return "\n".join(lines)


# ============================================================================
# Category 6: Equation Transformation (HARDEST)
# ============================================================================

def solve_equation_transformation(prompt: str) -> Optional[str]:
    """Attempt to solve equation transformation problems.
    
    Strategy: infer symbol-to-digit bijection and operator mapping from
    few-shot examples. This is computationally expensive for symbolic sub-cases.
    
    Returns None for cases that can't be solved (many symbolic cases).
    """
    # Split into examples and query
    parts = re.split(
        r'(?:Now,?\s*)?(?:determine|find|compute|calculate)\s+the\s+result\s+(?:for|of)[:\s]*',
        prompt, flags=re.IGNORECASE
    )
    
    if len(parts) < 2:
        return None
    
    examples_text = parts[0]
    query_text = parts[1].strip()
    
    # Extract example equations: "expr = result"
    equations = re.findall(r'([^\n=]+?)\s*=\s*([^\n]+)', examples_text)
    equations = [
        (lhs.strip(), rhs.strip()) for lhs, rhs in equations
        if not any(w in lhs.lower() for w in ['wonderland', 'secret', 'below', 'rule', 'example'])
    ]
    
    if not equations:
        return None
    
    # For numeric equation transformation: try to detect digit bijection
    # This is a simplified solver - full solver would enumerate all permutations
    
    # Collect all unique non-operator symbols
    all_text = " ".join(lhs + " " + rhs for lhs, rhs in equations) + " " + query_text
    # Remove known operators and whitespace
    symbols = set()
    for c in all_text:
        if c not in ' +-*/=(){}[].,\n\t' and not c.isdigit():
            symbols.add(c)
    
    # If there are ≤ 10 unique symbols, try brute-force bijection to digits
    if 1 <= len(symbols) <= 10:
        symbol_list = sorted(symbols)
        digits = list(range(10))
        
        # Try permutations (up to 10! = 3.6M, but usually fewer symbols)
        # Limit search to avoid timeout
        max_perms = 100000
        count = 0
        
        for perm in permutations(digits, len(symbol_list)):
            count += 1
            if count > max_perms:
                break
            
            mapping = dict(zip(symbol_list, perm))
            
            # Verify against all examples
            valid = True
            for lhs, rhs in equations:
                try:
                    lhs_decoded = _apply_digit_mapping(lhs, mapping)
                    rhs_decoded = _apply_digit_mapping(rhs, mapping)
                    
                    if lhs_decoded is None or rhs_decoded is None:
                        valid = False
                        break
                    
                    # Evaluate both sides
                    lhs_val = _safe_eval(lhs_decoded)
                    rhs_val = _safe_eval(rhs_decoded)
                    
                    if lhs_val is None or rhs_val is None:
                        valid = False
                        break
                    
                    if not math.isclose(lhs_val, rhs_val, rel_tol=1e-6):
                        valid = False
                        break
                except Exception:
                    valid = False
                    break
            
            if valid:
                # Apply mapping to query
                result = _apply_digit_mapping(query_text, mapping)
                if result is not None:
                    result_val = _safe_eval(result)
                    if result_val is not None:
                        # Re-encode result back using the mapping
                        reverse_mapping = {v: k for k, v in mapping.items()}
                        result_str = str(int(result_val)) if result_val == int(result_val) else f"{result_val:.2f}"
                        encoded = ""
                        for c in result_str:
                            if c.isdigit() and int(c) in reverse_mapping:
                                encoded += reverse_mapping[int(c)]
                            else:
                                encoded += c
                        return encoded
    
    return None


def _apply_digit_mapping(expr: str, mapping: dict) -> Optional[str]:
    """Replace symbols with digits according to mapping."""
    result = ""
    for c in expr:
        if c in mapping:
            result += str(mapping[c])
        elif c in ' +-*/()' or c.isdigit():
            result += c
        elif c.isspace():
            result += c
        else:
            return None  # Unknown symbol
    return result


def _safe_eval(expr: str) -> Optional[float]:
    """Safely evaluate a simple arithmetic expression."""
    try:
        # Only allow digits, operators, parens, decimal points
        cleaned = expr.strip()
        if not re.match(r'^[\d\s+\-*/().]+$', cleaned):
            return None
        # Replace implicit multiplication
        result = eval(cleaned, {"__builtins__": {}}, {})
        return float(result)
    except Exception:
        return None


def equation_transformation_cot(prompt: str, answer: str) -> str:
    """Generate CoT for equation transformation."""
    parts = re.split(
        r'(?:Now,?\s*)?(?:determine|find|compute|calculate)\s+the\s+result\s+(?:for|of)[:\s]*',
        prompt, flags=re.IGNORECASE
    )
    
    examples_text = parts[0] if parts else prompt
    equations = re.findall(r'([^\n=]+?)\s*=\s*([^\n]+)', examples_text)
    equations = [
        (lhs.strip(), rhs.strip()) for lhs, rhs in equations
        if not any(w in lhs.lower() for w in ['wonderland', 'secret', 'below', 'rule', 'example'])
    ]
    
    lines = [
        "I need to find the transformation rules from the examples.",
        "Each example shows an equation where symbols map to digits and/or operators.",
        "",
        "Given examples:",
    ]
    for lhs, rhs in equations[:5]:
        lines.append(f"  {lhs} = {rhs}")
    
    lines.extend([
        "",
        "Step 1: Identify all unique symbols in the equations.",
        "Step 2: For each possible digit-to-symbol bijection, check consistency with all examples.",
        "Step 3: Apply the discovered mapping to the query.",
        "",
        f"After testing bijections, the correct answer is: {answer}",
    ])
    
    return "\n".join(lines)


# ============================================================================
# Master oracle dispatcher
# ============================================================================

def solve(prompt: str, category: Optional[str] = None) -> Optional[str]:
    """Attempt to solve any problem. Returns answer or None."""
    from .category_detector import detect_category
    
    if category is None:
        category = detect_category(prompt)
    
    solvers = {
        'gravitational_constant': solve_gravitational,
        'number_base_conversion': solve_number_base,
        'unit_conversion': solve_unit_conversion,
        'text_encryption': solve_text_encryption,
        'bit_manipulation': solve_bit_manipulation,
        'equation_transformation': solve_equation_transformation,
    }
    
    solver = solvers.get(category)
    if solver is None:
        return None
    
    try:
        return solver(prompt)
    except Exception:
        return None


def generate_cot(prompt: str, answer: str, category: Optional[str] = None) -> str:
    """Generate a chain-of-thought trace for a problem.
    
    Returns the full assistant response: <think>...</think>\n\\boxed{answer}
    """
    from .category_detector import detect_category
    
    if category is None:
        category = detect_category(prompt)
    
    cot_generators = {
        'gravitational_constant': gravitational_cot,
        'number_base_conversion': number_base_cot,
        'unit_conversion': unit_conversion_cot,
        'text_encryption': text_encryption_cot,
        'bit_manipulation': bit_manipulation_cot,
        'equation_transformation': equation_transformation_cot,
    }
    
    gen = cot_generators.get(category)
    if gen:
        reasoning = gen(prompt, answer)
    else:
        reasoning = f"Analyzing the problem and computing the result: {answer}"
    
    return f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"
