"""
Character Validation Module: Strict Scanned Character Enforcement
Ensures that generation only proceeds if 100% of required characters have been
scanned, processed, and smartly understood. Synthetic fallbacks are prohibited.
"""

from typing import Dict, List, Set, Tuple


class MissingCharactersError(Exception):
    """Raised when text contains characters not present in the scanned glyph bank."""
    def __init__(self, missing_characters: List[str], available_count: int):
        self.missing_characters = missing_characters
        self.available_count = available_count
        chars_str = ", ".join(repr(c) for c in missing_characters)
        super().__init__(
            f"Cannot generate assignment: The following characters have not been scanned "
            f"and understood yet: [{chars_str}]. Please scan notes containing these characters first."
        )


def get_required_characters(text: str) -> Set[str]:
    """
    Extracts all unique characters in the document that require a glyph,
    skipping standard whitespace (spaces, tabs, newlines) and markdown syntax markers.
    """
    whitespace = {" ", "\t", "\n", "\r"}
    required = set()

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # Strip markdown headings
        if line.startswith("### "):
            line = line[4:]
        elif line.startswith("## "):
            line = line[3:]
        elif line.startswith("# "):
            line = line[2:]

        # Strip markdown bullets (rendered as →)
        if line.startswith("→ ") or line.startswith("-> "):
            required.add("→")
            line = line[2:] if line.startswith("→ ") else line[3:]
        elif line.startswith("- ") or line.startswith("* "):
            required.add("→")
            line = line[2:]

        for ch in line:
            if ch not in whitespace:
                required.add(ch)

    return required


def validate_text_coverage(
    text: str,
    glyph_bank: Dict
) -> Tuple[bool, List[str], List[str], float]:
    """
    Validates whether all characters in the text exist in the scanned glyph bank.

    Args:
        text: Input assignment text to validate.
        glyph_bank: Dictionary mapping character labels to lists of glyph variants.

    Returns:
        (is_valid, missing_characters, available_characters, coverage_percentage)
    """
    required = get_required_characters(text)
    if not required:
        return True, [], sorted(list(glyph_bank.keys())), 100.0

    # A character is valid if it is in glyph_bank and has at least one variant
    scanned_chars = set(
        ch for ch, variants in glyph_bank.items()
        if variants and len(variants) > 0
    )

    missing = sorted(list(required - scanned_chars))
    present = sorted(list(required.intersection(scanned_chars)))

    coverage_pct = round((len(present) / len(required)) * 100.0, 1)
    is_valid = len(missing) == 0

    return is_valid, missing, sorted(list(scanned_chars)), coverage_pct
