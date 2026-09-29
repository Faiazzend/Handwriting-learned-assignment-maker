"""
Word Validator Engine: Validates and corrects OCR-extracted words before entry
into the cursive whole-word bank.

Prevents OCR misread garbage (e.g. 'pesple', 'agnee', 'comtnact', 'pesplerufenprat')
from polluting the authentic word bank while intelligently rescuing slightly misread words
via fuzzy alignment to known note vocabulary.
"""

import re
from difflib import SequenceMatcher

# Core vocabulary extracted directly from ground-truth scanned note pages 1, 2, 3
GROUND_TRUTH_VOCAB = {
    "a", "about", "abstract", "accept", "administrative", "advantage", "after", "again",
    "against", "agree", "agreement", "all", "alternative", "an", "and", "are", "as",
    "at", "attempt", "authority", "be", "because", "been", "being", "between", "bind",
    "body", "bureaucracy", "bureaucratic", "but", "by", "came", "can", "center",
    "centralized", "challenge", "citizen", "citizens", "citizenship", "claims", "coercion",
    "collective", "collectively", "comes", "complete", "concentrated", "concept", "conflict",
    "consolidation", "contained", "contract", "core", "could", "creation", "days", "decision",
    "deeply", "deliberate", "demonstrating", "dependent", "different", "differently", "didn't",
    "do", "document", "early", "effect", "efforts", "emerged", "established", "europe",
    "every", "everyone", "explains", "fear", "feature", "first", "for", "form",
    "foundation", "fragmented", "freedom", "freedoms", "from", "fundamentally", "gain",
    "gained", "governance", "governing", "government", "gradually", "grew", "had", "has",
    "have", "he", "historic", "history", "idea", "in", "institutions", "interpret",
    "into", "is", "it", "it's", "its", "kept", "legitimacy", "legitimate", "lies",
    "limits", "live", "made", "management", "many", "mixture", "model", "modern",
    "monopolizes", "more", "nature", "networks", "no", "not", "obedience", "obey",
    "of", "on", "one", "only", "order", "organization", "organize", "organized",
    "origin", "originate", "other", "our", "over", "over-time", "people", "philosophical",
    "places", "political", "population", "power", "practical", "product", "provides",
    "punishment", "reality", "settled", "simply", "social", "society", "stability",
    "stable", "state", "struggle", "struggled", "sustain", "systems", "taxations",
    "territory", "that", "the", "their", "them", "then", "theory", "therefore",
    "these", "they", "things", "this", "those", "thus", "tied", "time", "to",
    "tools", "turned", "under", "up", "upper-hand", "use", "violence", "war",
    "war-making", "was", "we", "were", "what", "when", "which", "who", "why",
    "will", "with", "without", "would"
}

# Regex for structurally valid single word (letters, optional single hyphen or apostrophe)
VALID_WORD_RE = re.compile(r"^[A-Za-z]+([-\'][A-Za-z]+)?$")

# Patterns that indicate OCR garble (e.g. 5+ consecutive consonants)
CONSONANT_CLUSTER_RE = re.compile(r"[bcdfghjklmnpqrstvwxz]{5,}", re.IGNORECASE)


def validate_and_normalize_word(raw_label, confidence=0.70):
    """
    Validates an OCR-detected word label.
    1. Rejects multi-word strings, internal punctuation, or strange characters.
    2. Rescues minor OCR misreads by fuzzy matching against ground-truth note vocabulary.
    3. Rejects garbled strings (4+ consecutive consonants, length extremes).

    Returns:
        (is_valid: bool, normalized_label: str or None)
    """
    if not raw_label or not isinstance(raw_label, str):
        return False, None

    clean = raw_label.strip(" \t\n\r.,;:!?\"'()[]{}")
    if not clean or len(clean) < 2 or len(clean) > 22:
        return False, None

    # Must match single word pattern (no internal spaces, colons, dots, etc.)
    if not VALID_WORD_RE.match(clean):
        return False, None

    lower = clean.lower()

    # Exact match in ground truth vocabulary
    if lower in GROUND_TRUTH_VOCAB:
        # Preserve original capitalization if title case, else lower
        return True, clean

    # Check for unpronounceable consonant clusters (common in OCR misread cursive)
    # y is treated as vowel (e.g. system, rhythm, type)
    if CONSONANT_CLUSTER_RE.search(lower):
        allowed_clusters = {"struggl", "length", "rhythm", "bstr"}
        if not any(ac in lower for ac in allowed_clusters):
            return False, None

    # Fuzzy match against ground truth vocabulary for minor OCR misreads
    # e.g., 'agnee' -> 'agree', 'comtnact' -> 'contract', 'pesple' -> 'people'
    best_match = None
    best_sim = 0.0

    for candidate in GROUND_TRUTH_VOCAB:
        # Quick length filter: diff length at most 2
        if abs(len(candidate) - len(lower)) > 2:
            continue
        sim = SequenceMatcher(None, lower, candidate).ratio()
        if sim > best_sim:
            best_sim = sim
            best_match = candidate

    # High similarity threshold for fuzzy rescue (>= 0.74 similarity)
    if best_match and best_sim >= 0.74:
        # Match casing
        corrected = best_match.capitalize() if clean[0].isupper() else best_match
        return True, corrected

    # If confidence is high (>= 0.85) and word has valid English vowels and syllable structure
    if confidence >= 0.85 and any(v in lower for v in "aeiouy"):
        return True, clean

    return False, None
