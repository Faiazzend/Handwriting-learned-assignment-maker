"""
Glyph Quality Gate: Validates that extracted crops are genuine single characters.
Rejects multi-letter fragments, noise specks, and corrupt blobs before they
enter the glyph bank.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

# ── Expected character dimensions at typical scan resolution (~150-200 DPI) ──
# These are rough bounding-box expectations for single characters in handwriting.
# Format: (min_w, max_w, min_h, max_h, min_ar, max_ar)

CHAR_EXPECTATIONS = {
    # Narrow x-height letters
    'i': (3, 18, 8, 45, 0.15, 1.0),
    'l': (3, 18, 10, 50, 0.10, 0.8),
    't': (3, 22, 8, 50, 0.15, 1.2),
    'f': (3, 22, 10, 50, 0.15, 1.2),
    'r': (3, 22, 6, 40, 0.20, 1.5),

    # Medium x-height letters
    'a': (5, 28, 6, 35, 0.30, 1.8),
    'c': (4, 25, 6, 35, 0.30, 1.8),
    'e': (4, 25, 6, 35, 0.30, 1.8),
    'n': (5, 30, 6, 35, 0.30, 1.8),
    'o': (4, 25, 6, 35, 0.30, 1.8),
    's': (4, 25, 6, 35, 0.25, 1.8),
    'u': (5, 28, 6, 35, 0.30, 1.8),
    'v': (5, 28, 6, 35, 0.35, 1.8),
    'x': (4, 28, 6, 35, 0.30, 1.8),
    'z': (4, 28, 6, 35, 0.30, 1.8),

    # Wide x-height letters
    'm': (8, 45, 6, 35, 0.60, 2.5),
    'w': (8, 42, 6, 35, 0.60, 2.5),

    # Ascender letters (taller)
    'b': (4, 25, 10, 50, 0.20, 1.2),
    'd': (4, 25, 10, 50, 0.20, 1.2),
    'h': (5, 28, 10, 50, 0.25, 1.3),
    'k': (5, 28, 10, 50, 0.25, 1.3),

    # Descender letters
    'g': (5, 28, 10, 50, 0.25, 1.3),
    'p': (5, 28, 10, 50, 0.25, 1.3),
    'y': (4, 28, 10, 50, 0.20, 1.3),
    'j': (3, 18, 10, 50, 0.10, 0.8),
    'q': (5, 28, 10, 50, 0.25, 1.3),

    # Capitals (generally larger)
    'A': (8, 40, 12, 55, 0.35, 1.5),
    'B': (8, 35, 12, 55, 0.30, 1.3),
    'C': (8, 35, 12, 55, 0.30, 1.5),
    'D': (8, 38, 12, 55, 0.30, 1.5),
    'E': (6, 32, 12, 55, 0.25, 1.3),
    'F': (6, 30, 12, 55, 0.25, 1.3),
    'G': (8, 38, 12, 55, 0.30, 1.5),
    'H': (8, 38, 12, 55, 0.35, 1.5),
    'I': (3, 22, 12, 55, 0.10, 1.0),
    'J': (4, 25, 12, 55, 0.15, 1.0),
    'K': (8, 38, 12, 55, 0.30, 1.5),
    'L': (6, 32, 12, 55, 0.25, 1.3),
    'M': (10, 45, 12, 55, 0.45, 2.0),
    'N': (8, 38, 12, 55, 0.35, 1.5),
    'O': (8, 38, 12, 55, 0.35, 1.5),
    'P': (6, 32, 12, 55, 0.25, 1.3),
    'Q': (8, 38, 12, 55, 0.35, 1.5),
    'R': (8, 35, 12, 55, 0.30, 1.3),
    'S': (6, 30, 10, 55, 0.25, 1.3),
    'T': (6, 35, 12, 55, 0.25, 1.5),
    'U': (8, 35, 12, 55, 0.35, 1.5),
    'V': (8, 35, 12, 55, 0.35, 1.5),
    'W': (10, 45, 12, 55, 0.50, 2.0),
    'X': (8, 35, 12, 55, 0.30, 1.5),
    'Y': (6, 35, 12, 55, 0.25, 1.5),
    'Z': (6, 32, 12, 55, 0.25, 1.3),

    # Punctuation
    '.': (2, 10, 2, 10, 0.3, 3.0),
    ',': (2, 12, 3, 15, 0.2, 2.0),
    '-': (4, 20, 2, 8, 1.0, 8.0),
    ':': (2, 10, 5, 20, 0.15, 1.0),
    ';': (2, 10, 5, 20, 0.15, 1.0),
    '→': (12, 60, 6, 30, 1.0, 5.0),
}

# Generic fallback for unknown characters
DEFAULT_EXPECTATIONS = (3, 45, 5, 55, 0.15, 3.0)


def compute_ink_density(crop_binary: np.ndarray) -> float:
    """Ratio of ink pixels to total bounding box pixels."""
    if crop_binary is None or crop_binary.size == 0:
        return 0.0
    total = crop_binary.shape[0] * crop_binary.shape[1]
    ink = np.count_nonzero(crop_binary)
    return ink / max(total, 1)


def compute_connected_components(crop_binary: np.ndarray) -> int:
    """Count the number of distinct connected components in a binary crop."""
    if crop_binary is None or crop_binary.size == 0:
        return 0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(crop_binary, connectivity=8)
    # Subtract 1 for background label
    # Filter out tiny components (noise dots < 6px area)
    significant = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 6:
            significant += 1
    return significant


def is_valid_single_glyph(
    crop_binary: np.ndarray,
    char_label: str,
    strict: bool = True
) -> Tuple[bool, str]:
    """
    Validates that a binary crop is a genuine single character glyph.
    
    Returns: (is_valid, rejection_reason)
    """
    if crop_binary is None or len(crop_binary.shape) < 2:
        return False, "null_or_invalid_shape"

    h, w = crop_binary.shape[:2]

    # Basic minimum size
    if w < 2 or h < 2:
        return False, f"too_small_{w}x{h}"

    # Check ink density
    density = compute_ink_density(crop_binary)
    if density < 0.08:
        return False, f"low_ink_density_{density:.2f}"
    if density > 0.92:
        return False, f"high_ink_density_{density:.2f}_likely_blob"

    # Get character-specific expectations
    expectations = CHAR_EXPECTATIONS.get(char_label, DEFAULT_EXPECTATIONS)
    min_w, max_w, min_h, max_h, min_ar, max_ar = expectations

    ar = w / max(h, 1)

    if strict:
        # Size validation
        if w > max_w * 1.5:
            return False, f"too_wide_{w}px_max_{max_w}_for_{repr(char_label)}"
        if h > max_h * 1.3:
            return False, f"too_tall_{h}px_max_{max_h}_for_{repr(char_label)}"

        # Aspect ratio validation (most reliable single-char indicator)
        if ar > max_ar * 1.3:
            return False, f"ar_{ar:.2f}_exceeds_max_{max_ar}_for_{repr(char_label)}_likely_multi_char"

    # Connected components check — a single letter should have 1-3 components
    # (e.g., 'i' has dot + body = 2, 'j' has dot + body = 2)
    # More than 4 significant components strongly suggests multi-letter fragment
    num_cc = compute_connected_components(crop_binary)
    if num_cc > 4:
        return False, f"too_many_components_{num_cc}_likely_multi_char"

    # For punctuation, looser rules
    if char_label in '.,;:-':
        if w > 25 or h > 25:
            return False, f"punctuation_too_large_{w}x{h}"
        return True, "ok"

    # For regular letters, at least minimum dimensions
    if char_label.isalpha():
        if w < 3 or h < 5:
            return False, f"letter_too_small_{w}x{h}"

    return True, "ok"


def pick_best_variant(variants: list, char_label: str) -> Optional[dict]:
    """
    From a list of glyph variants, pick the one with the best quality score.
    Returns the variant dict or None if all are invalid.
    """
    import os

    scored = []
    expectations = CHAR_EXPECTATIONS.get(char_label, DEFAULT_EXPECTATIONS)
    _, _, _, _, _, _ = expectations

    for v in variants:
        path = v.get("path")
        if not path or not os.path.exists(path):
            continue

        w, h = v.get("width", 0), v.get("height", 0)
        if w < 3 or h < 3:
            continue

        ar = w / max(h, 1)

        # Score: prefer variants close to expected proportions
        # Lower score = better
        score = 0

        # Penalize extreme aspect ratios
        expected_ar = (expectations[4] + expectations[5]) / 2
        score += abs(ar - expected_ar) * 10

        # Penalize very small or very large
        expected_h = (expectations[2] + expectations[3]) / 2
        score += abs(h - expected_h) / expected_h * 5

        scored.append((score, v))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0])
    return scored[0][1]
