"""
Unruled Page Layout Engine — target-height normalized handwriting assembly.

Every word is produced by the first strategy that succeeds:

  Case 1   Authentic scanned whole word (exact label match, crop sanity-checked),
           normalized to a target height derived from its letters.
  Case 1.5 VATr neural synthesis — only when the synthesizer is Tier 1 with a style
           embedding, and only if the generated image passes an ink-quality gate.
  Case 2   Character-by-character assembly from the scanned glyph bank. Each glyph
           is normalized to its typographic class height (x-height / ascender /
           descender) instead of being multiplied by a flat factor.

Robustness:
- Per-character outlier variants (mis-cropped fragments, merged letters) are rejected
  by comparing each variant's height/aspect against that character's median.
- Whole-word crops whose width-per-letter or height is implausible for their label
  (mis-aligned OCR labels, multi-line crops) are excluded from the word bank.

STRICT: never renders a synthetic computer font. Any un-scanned character raises
MissingCharactersError.
"""

import os
import json
import random
import logging
import statistics

import numpy as np
from PIL import Image

from engine.validator import validate_text_coverage, MissingCharactersError
from engine.stroke_augmentor import SmartVariantSelector

logger = logging.getLogger(__name__)

# ── Typographical character classifications ──────────────────────────
X_HEIGHT_CHARS = set("acemnorsuvwxz")
ASCENDER_CHARS = set("bdfhkltABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?/()[]{}&")
DESCENDER_CHARS = set("gjpqy")
PUNCTUATION_BASELINE = set(".,")
PUNCTUATION_MID = set("-:;=")
PUNCTUATION_HIGH = set("'\"^")
ARROW_CHARS = set("\u2192")

# ── Normalization constants (A4 @ 300 DPI) ───────────────────────────
BASE_X_HEIGHT = 35            # px height of x-height letters (a, c, e ...) ≈ 3 mm
TALL_RATIO = 1.6              # ascender / descender letters are 1.6 × x-height
MAX_SCALE_DEVIATION = 2.0     # a glyph's scale may differ from the bank's natural scale by at most 2×
MAX_GLYPH_WIDTH_RATIO = 2.4   # widest sane letter (M, W) relative to x-height
MAX_ARROW_WIDTH_RATIO = 3.5

# Variant outlier rejection (relative to the character's median)
VARIANT_HEIGHT_RANGE = (0.65, 1.55)
VARIANT_ASPECT_RANGE = (0.5, 2.0)

# Whole-word crop sanity (relative to bank-wide medians)
WORD_WIDTH_PER_CHAR_RANGE = (0.55, 1.8)
WORD_HEIGHT_RANGE = (0.55, 2.1)

# Neural synthesis quality gate
NEURAL_INK_COVERAGE_RANGE = (0.03, 0.45)   # handwriting is sparse; a solid block is ~0.9
NEURAL_MIN_ALPHA_STD = 0.12
NEURAL_WIDTH_RATIO_RANGE = (0.4, 2.5)
NEURAL_MAX_CONSECUTIVE_FAILURES = 3

# path -> (mtime, bbox-cropped RGBA image). Shared across layouts; images are never mutated.
_CROP_CACHE = {}


def _load_cropped(path):
    """Load a glyph/word PNG as RGBA, cropped to its ink bounding box (cached by mtime)."""
    mtime = os.path.getmtime(path)
    hit = _CROP_CACHE.get(path)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    img = Image.open(path).convert("RGBA")
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    _CROP_CACHE[path] = (mtime, img)
    return img


def _upscale_smooth(img, new_w, new_h):
    """
    Resize an ink RGBA image and, when upscaling, re-form crisp but smooth stroke edges.

    The scanned glyphs are tiny (x-height ≈ 10 px), so a plain 3–4× upscale yields
    blocky stair-stepped strokes. Blurring the alpha proportionally to the scale and
    re-applying a soft threshold produces pen-like contours without thinning strokes.
    """
    from PIL import ImageFilter
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    scale = new_h / float(max(img.height, 1))
    if scale < 1.5:
        return resized
    r, g, b, a = resized.split()
    a = a.filter(ImageFilter.GaussianBlur(radius=scale * 0.35))
    lo, hi = 70, 170   # soft threshold band → anti-aliased edge ~1–2 px wide
    a = a.point(lambda v: 0 if v <= lo else (255 if v >= hi else int((v - lo) * 255 / (hi - lo))))
    return Image.merge("RGBA", (r, g, b, a))


def _word_height_units(text):
    """Height of a word in x-height units: 1 + ascender extension + descender extension."""
    has_asc = any(c in ASCENDER_CHARS for c in text)
    has_desc = any(c in DESCENDER_CHARS for c in text)
    ext = TALL_RATIO - 1.0
    return 1.0 + (ext if has_asc else 0.0) + (ext if has_desc else 0.0), has_desc


class ProportionalGlyphScaler:
    """
    Target-height glyph normalizer.

    sf = target_height / original_height, where the target comes from the glyph's
    typographic class. Punctuation and symbols (no class target) keep the writer's
    natural size via the bank-derived global scale. The per-glyph scale is bounded
    to within MAX_SCALE_DEVIATION of the global scale so a mis-cropped fragment can
    never be blown up into a giant blob.
    """

    def __init__(self, glyph_bank=None, base_scale=3.0, base_x_height=BASE_X_HEIGHT):
        self.base_x_height = base_x_height
        self.global_scale = base_scale        # fallback until the bank is analyzed
        self.scan_x_height = None             # writer's x-height in scan pixels
        self.char_stats = {}                  # char -> {"median_h", "median_aspect", "n"}
        if glyph_bank:
            self._analyze_bank(glyph_bank)

    def _analyze_bank(self, glyph_bank):
        for ch, variants in glyph_bank.items():
            dims = []
            for v in variants or []:
                p = v.get("path")
                if not p or not os.path.exists(p):
                    continue
                try:
                    img = _load_cropped(p)
                except Exception:
                    continue
                if img.width > 0 and img.height > 0:
                    dims.append((img.width, img.height))
            if dims:
                self.char_stats[ch] = {
                    "median_h": statistics.median(h for _, h in dims),
                    "median_aspect": statistics.median(w / h for w, h in dims),
                    "n": len(dims),
                }

        xh = [s["median_h"] for c, s in self.char_stats.items() if c in X_HEIGHT_CHARS]
        if xh:
            self.scan_x_height = statistics.median(xh)
            if self.scan_x_height > 0:
                self.global_scale = self.base_x_height / self.scan_x_height

    def target_height(self, char, extra_scale=1.0):
        """Class target height in px, or None for punctuation/symbols (natural size)."""
        if char in ASCENDER_CHARS or char in DESCENDER_CHARS:
            return self.base_x_height * TALL_RATIO * extra_scale
        if char in X_HEIGHT_CHARS or char == "i":
            return self.base_x_height * extra_scale
        return None

    def scale_factor(self, char, width, height, extra_scale=1.0):
        natural = self.global_scale * extra_scale
        target = self.target_height(char, extra_scale)
        if target is None:
            sf = natural
        else:
            sf = target / float(height)
            sf = min(max(sf, natural / MAX_SCALE_DEVIATION), natural * MAX_SCALE_DEVIATION)

        width_ratio = MAX_ARROW_WIDTH_RATIO if char in ARROW_CHARS else MAX_GLYPH_WIDTH_RATIO
        max_w = width_ratio * self.base_x_height * extra_scale
        if width * sf > max_w:
            sf = max_w / float(width)
        return sf

    def scale_glyph(self, char, raw_img, extra_scale=1.0):
        bbox = raw_img.getbbox()
        img = raw_img.crop(bbox) if bbox else raw_img
        w, h = img.size
        if w == 0 or h == 0:
            return img, 0

        sf = self.scale_factor(char, w, h, extra_scale)
        new_w = max(int(round(w * sf)), 2)
        new_h = max(int(round(h * sf)), 2)
        resized = _upscale_smooth(img, new_w, new_h)

        y_offset = self._compute_baseline_offset(char, new_h, extra_scale)
        return resized, y_offset

    def _compute_baseline_offset(self, char, glyph_height, extra_scale=1.0):
        """Offset of the glyph's top edge relative to the baseline (negative = above)."""
        xh = self.base_x_height * extra_scale
        if char in DESCENDER_CHARS:
            # x-height body sits above the baseline, tail hangs below
            return -int(round(glyph_height / TALL_RATIO))
        if char in (",", ";"):
            # tail dips slightly below the baseline
            return -int(round(glyph_height * 0.6))
        if char in ("-", "="):
            # centred on the middle of the x-height band
            return -int(round(xh * 0.5 + glyph_height * 0.5))
        if char in PUNCTUATION_HIGH:
            # hangs from the ascender line
            return -int(round(xh * TALL_RATIO))
        if char in ARROW_CHARS:
            return -int(round(xh * 0.5 + glyph_height * 0.5))
        # x-height letters, ascenders, capitals, digits, '.', ':' — bottom on the baseline
        return -glyph_height


class UnruledPageLayout:
    def __init__(self, glyph_bank=None, synthesizer=None, words_index_path=None, use_neural=True):
        self.glyph_bank = glyph_bank or {}
        self.synthesizer = synthesizer
        self.use_neural = use_neural

        # A4 at 300 DPI
        self.page_width = 2480
        self.page_height = 3508

        # Margins & spacing
        self.base_left_margin = 220
        self.base_right_margin = 220
        self.base_top_margin = 280
        self.base_bottom_margin = 280
        self.base_line_height = 110

        # Target-height normalizer, calibrated from the writer's own glyph bank
        self.scaler = ProportionalGlyphScaler(self.glyph_bank, base_scale=3.0)

        # N-gram aware variant selector — prevents visible repetition
        self.variant_selector = SmartVariantSelector(history_depth=3)

        self._variant_cache = {}
        self._neural_failures = 0
        self._neural_disabled = False
        self.quality_report = {
            "rejected_glyph_variants": 0,
            "rejected_word_crops": 0,
            "word_bank_entries": 0,
            "neural_words": 0,
            "neural_rejected": 0,
        }

        # Authentic whole-word bank (exact label -> list of entries)
        self.word_bank = {}
        if words_index_path and os.path.exists(words_index_path):
            self._load_word_bank(words_index_path)

    # ── Banks ────────────────────────────────────────────────────────

    def _load_word_bank(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception as e:
            logger.warning(f"Word bank unreadable ({path}): {e}")
            return

        measured = []
        for item in entries:
            label = (item.get("label") or "").strip()
            p = item.get("path")
            if not label or not p or not os.path.exists(p):
                continue
            try:
                img = _load_cropped(p)
            except Exception:
                continue
            if img.width > 0 and img.height > 0:
                measured.append((label, item, img.width, img.height))
        if not measured:
            return

        median_ppc = statistics.median(w / len(lbl) for lbl, _, w, _ in measured)
        scan_xh = self.scaler.scan_x_height or statistics.median(
            h / _word_height_units(lbl)[0] for lbl, _, _, h in measured
        )

        for label, item, w, h in measured:
            width_ratio = (w / len(label)) / median_ppc
            height_ratio = h / (scan_xh * _word_height_units(label)[0])
            if (WORD_WIDTH_PER_CHAR_RANGE[0] <= width_ratio <= WORD_WIDTH_PER_CHAR_RANGE[1]
                    and WORD_HEIGHT_RANGE[0] <= height_ratio <= WORD_HEIGHT_RANGE[1]):
                self.word_bank.setdefault(label, []).append(item)
                self.quality_report["word_bank_entries"] += 1
            else:
                self.quality_report["rejected_word_crops"] += 1

    def _usable_variants(self, char):
        """Variants of `char` with valid files, excluding size/shape outliers."""
        cached = self._variant_cache.get(char)
        if cached is not None:
            return cached

        measured = []
        for v in self.glyph_bank.get(char, []) or []:
            p = v.get("path")
            if not p or not os.path.exists(p):
                continue
            try:
                img = _load_cropped(p)
            except Exception:
                continue
            if img.width > 0 and img.height > 0:
                measured.append((v, img.width, img.height))

        if len(measured) >= 3:
            med_h = statistics.median(h for _, _, h in measured)
            med_a = statistics.median(w / h for _, w, h in measured)
            keep = [
                v for v, w, h in measured
                if VARIANT_HEIGHT_RANGE[0] * med_h <= h <= VARIANT_HEIGHT_RANGE[1] * med_h
                and VARIANT_ASPECT_RANGE[0] * med_a <= (w / h) <= VARIANT_ASPECT_RANGE[1] * med_a
            ]
            if not keep:
                keep = [min(measured, key=lambda m: abs(m[2] - med_h))[0]]
        else:
            keep = [v for v, _, _ in measured]

        self.quality_report["rejected_glyph_variants"] += len(measured) - len(keep)
        self._variant_cache[char] = keep
        return keep

    def _get_glyph_for_char(self, char, last_variant_id=None):
        """
        Retrieves an authentic scanned glyph variant using N-gram aware selection.
        STRICT: Never creates synthetic fallbacks.
        """
        valid_variants = self._usable_variants(char)
        if not valid_variants:
            raise MissingCharactersError(missing_characters=[char], available_count=len(self.glyph_bank))

        selected, vid = self.variant_selector.select(char, valid_variants)
        if selected is None:
            selected = random.choice(valid_variants)
            vid = selected.get("variant_id")

        return _load_cropped(selected["path"]), vid

    # ── Neural synthesis (Case 1.5) ──────────────────────────────────

    def _seed_neural_style(self):
        """Give the synthesizer a style embedding from validated whole-word crops (once)."""
        syn = self.synthesizer
        refs = []
        for entries in self.word_bank.values():
            for item in entries:
                try:
                    crop = _load_cropped(item["path"])
                except Exception:
                    continue
                flat = Image.new("RGB", crop.size, (255, 255, 255))
                flat.paste(crop, mask=crop.split()[3])
                refs.append(flat)
                if len(refs) >= 15:
                    break
            if len(refs) >= 15:
                break
        if refs:
            try:
                syn.set_style_references(refs)
            except Exception as e:
                logger.warning(f"VATr style seeding failed: {e}")

    def _passes_neural_quality_gate(self, img, word):
        if img is None or img.width < 2 or img.height < 2:
            return False
        alpha = np.asarray(img.convert("RGBA").split()[3], dtype=np.float32) / 255.0
        coverage = float((alpha > 0.25).mean())
        if not (NEURAL_INK_COVERAGE_RANGE[0] <= coverage <= NEURAL_INK_COVERAGE_RANGE[1]):
            return False
        if float(alpha.std()) < NEURAL_MIN_ALPHA_STD:
            return False
        expected_w = len(word) * img.height / TALL_RATIO
        ratio = img.width / max(expected_w, 1.0)
        return NEURAL_WIDTH_RATIO_RANGE[0] <= ratio <= NEURAL_WIDTH_RATIO_RANGE[1]

    def _try_neural_word(self, clean_word, extra_scale):
        syn = self.synthesizer
        if not self.use_neural or self._neural_disabled or syn is None:
            return None

        # Check if synthesizer is HybridAIProvider or AIProvider with synthesize_word
        if hasattr(syn, "synthesize_word"):
            try:
                img = syn.synthesize_word(clean_word, target_height=int(55 * extra_scale))
                if img is not None and self._passes_neural_quality_gate(img, clean_word):
                    self._neural_failures = 0
                    self.quality_report["neural_words"] += 1
                    return img
            except Exception as e:
                logger.warning(f"AI Provider synthesis raised for '{clean_word}': {e}")
            return None

        if getattr(syn, "tier", 3) != 1 or getattr(syn, "model", None) is None:
            return None
        # VATr's content encoder only knows ASCII letters; skip symbols/bullets
        if not clean_word or not all(c.isascii() and (c.isalpha() or c in "'-") for c in clean_word):
            return None

        if getattr(syn, "style_embedding", None) is None:
            self._seed_neural_style()
            if getattr(syn, "style_embedding", None) is None:
                self._neural_disabled = True
                return None

        try:
            img = syn.generate_word(
                clean_word,
                target_height=int(55 * extra_scale),
                allow_procedural_fallback=False,
            )
        except Exception as e:
            logger.warning(f"VATr generation raised for '{clean_word}': {e}")
            img = None

        if img is not None and self._passes_neural_quality_gate(img, clean_word):
            self._neural_failures = 0
            self.quality_report["neural_words"] += 1
            return img

        self.quality_report["neural_rejected"] += 1
        self._neural_failures += 1
        if self._neural_failures >= NEURAL_MAX_CONSECUTIVE_FAILURES:
            self._neural_disabled = True
            logger.warning(
                "VATr output failed the ink-quality gate repeatedly; using authentic "
                "glyph assembly for the rest of this document."
            )
        return None

    # ── Word rendering ───────────────────────────────────────────────

    def _compose_whole_word(self, raw_img, word_text, trailing_punct, extra_scale, total_h, baseline_y):
        """Normalize a whole-word image to its target height and anchor it on the baseline."""
        bbox = raw_img.getbbox()
        if bbox:
            raw_img = raw_img.crop(bbox)
        if raw_img.width == 0 or raw_img.height == 0:
            return None

        units, has_desc = _word_height_units(word_text)
        xh = self.scaler.base_x_height * extra_scale
        target_nh = xh * units
        sf = target_nh / float(raw_img.height)
        nw = max(int(round(raw_img.width * sf)), 10)
        nh = max(int(round(raw_img.height * sf)), 10)
        resized = _upscale_smooth(raw_img, nw, nh)

        desc_depth = int(round(xh * (TALL_RATIO - 1.0))) if has_desc else 0
        dest_y = baseline_y + desc_depth - nh
        dest_y = max(0, min(total_h - nh, dest_y))

        punct_img, punct_y_off = None, 0
        if trailing_punct:
            try:
                p_raw, _ = self._get_glyph_for_char(trailing_punct[0])
                punct_img, punct_y_off = self.scaler.scale_glyph(trailing_punct[0], p_raw, extra_scale=extra_scale)
            except Exception:
                punct_img = None

        canvas_w = nw + (punct_img.width + 10 if punct_img else 20)
        word_canvas = Image.new("RGBA", (canvas_w, max(total_h, nh)), (0, 0, 0, 0))
        word_canvas.alpha_composite(resized, dest=(2, dest_y))

        if punct_img:
            p_dest_y = baseline_y + punct_y_off
            p_dest_y = max(0, min(total_h - punct_img.height, p_dest_y))
            word_canvas.alpha_composite(punct_img, dest=(nw + 3, p_dest_y))

        out_bbox = word_canvas.getbbox()
        if out_bbox:
            return word_canvas.crop((out_bbox[0], 0, out_bbox[2] + 2, total_h))
        return word_canvas

    def _generate_word_image(self, word, is_heading=False, heading_level=0):
        """
        Generates a word image:
        1.   Authentic scanned whole word (exact label, sanity-checked)
        1.5  VATr neural synthesis (Tier 1 only, quality-gated)
        2.   Target-height normalized character assembly from the glyph bank
        """
        extra_scale = 1.0
        if is_heading:
            extra_scale = 1.25 if heading_level == 1 else (1.15 if heading_level == 2 else 1.05)

        total_h = int(self.base_line_height * (1.3 if is_heading else 1.0))
        baseline_y = int(total_h * 0.72)

        clean_word = word.rstrip(".,;:!?'\"-")
        trailing_punct = word[len(clean_word):]

        # ── Case 1: authentic whole word (exact case — 'State' ≠ 'state') ──
        if clean_word:
            entries = [e for e in self.word_bank.get(clean_word, []) if os.path.exists(e.get("path", ""))]
            if entries:
                try:
                    raw = _load_cropped(random.choice(entries)["path"])
                    composed = self._compose_whole_word(raw, clean_word, trailing_punct, extra_scale, total_h, baseline_y)
                    if composed is not None:
                        return composed
                except Exception as e:
                    logger.debug(f"Word-bank render failed for '{clean_word}': {e}")

        # ── Case 1.5: VATr neural synthesis ──
        neural_img = self._try_neural_word(clean_word, extra_scale)
        if neural_img is not None:
            composed = self._compose_whole_word(neural_img, clean_word, trailing_punct, extra_scale, total_h, baseline_y)
            if composed is not None:
                return composed

        # ── Case 2: character-by-character assembly from authentic glyph bank ──
        char_items = []
        last_var = None
        for ch in word:
            raw_img, vid = self._get_glyph_for_char(ch, last_var)
            last_var = vid
            scaled_img, y_off = self.scaler.scale_glyph(ch, raw_img, extra_scale=extra_scale)
            char_items.append((ch, scaled_img, y_off))

        if not char_items:
            return Image.new("RGBA", (20, total_h), (0, 0, 0, 0))

        avg_width = sum(img.width for _, img, _ in char_items) / len(char_items)
        base_kern = max(2, int(avg_width * 0.12))

        total_w = sum(img.width for _, img, _ in char_items) + len(char_items) * (base_kern + 3) + 20
        word_canvas = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
        x_curr = 2

        # Progressive slant drift within word (simulates wrist angle shift)
        slant_drift = random.uniform(-0.15, 0.15)
        cumulative_slant = 0.0

        for idx, (ch, img, y_off) in enumerate(char_items):
            jitter_y = random.choice([-1, 0, 0, 1])
            cumulative_slant += slant_drift
            drift_y = int(cumulative_slant)

            dest_y = baseline_y + y_off + jitter_y + drift_y
            dest_y = max(0, min(total_h - img.height, dest_y))
            word_canvas.alpha_composite(img, dest=(x_curr, dest_y))

            # Variable kerning — tighter after narrow chars, wider after wide chars
            kern = base_kern + random.choice([-1, 0, 0, 0, 1])
            if ch in "ilt.,":
                kern = max(1, kern - 2)
            elif ch in "mwMW":
                kern += 2
            if idx + 1 < len(char_items):
                next_ch = char_items[idx + 1][0]
                if (ch, next_ch) in [('a', 't'), ('i', 'n'), ('e', 'r'), ('t', 'h'), ('o', 'n'), ('a', 'n'), ('e', 'n')]:
                    kern = max(0, kern - 1)

            x_curr += img.width + kern

        bbox = word_canvas.getbbox()
        if bbox:
            return word_canvas.crop((bbox[0], 0, bbox[2] + 2, total_h))
        return word_canvas

    # ── Page layout ──────────────────────────────────────────────────

    def layout_document(self, text, baseline_slant=0.4, margin_drift=12, line_spacing=95):
        """
        Validates coverage, parses text, places words onto A4 pages.
        Raises MissingCharactersError if any character is un-scanned.
        """
        is_valid, missing, avail, pct = validate_text_coverage(text, self.glyph_bank)
        if not is_valid:
            raise MissingCharactersError(missing_characters=missing, available_count=len(avail))

        pages = []
        current_page_glyphs = []
        current_y = self.base_top_margin
        current_left_margin = self.base_left_margin + random.uniform(-6, 6)

        for raw_line in text.split("\n"):
            line = raw_line.strip()

            # Blank line = paragraph break
            if not line:
                current_y += int(line_spacing * 0.7)
                if current_y > self.page_height - self.base_bottom_margin:
                    pages.append(current_page_glyphs)
                    current_page_glyphs = []
                    current_y = self.base_top_margin
                continue

            # Detect headings
            is_heading = False
            heading_level = 0
            if line.startswith("### "):
                is_heading, heading_level, line = True, 3, line[4:]
            elif line.startswith("## "):
                is_heading, heading_level, line = True, 2, line[3:]
            elif line.startswith("# "):
                is_heading, heading_level, line = True, 1, line[2:]

            # Detect bullet points & arrows
            is_bullet = False
            if line.startswith("\u2192 ") or line.startswith("-> "):
                is_bullet = True
                line = "\u2192 " + (line[2:] if line.startswith("\u2192 ") else line[3:])
            elif line.startswith("- ") or line.startswith("* "):
                is_bullet = True
                line = "\u2192 " + line[2:]

            words = [w for w in line.split(" ") if w]

            # Line baseline angle
            line_angle_rad = np.radians(baseline_slant + random.uniform(-0.2, 0.2))

            cursor_x = current_left_margin + (35 if is_bullet else 0)

            # Waver margin organically
            current_left_margin += random.uniform(-margin_drift * 0.1, margin_drift * 0.1)
            current_left_margin = max(
                self.base_left_margin - 20,
                min(self.base_left_margin + 30, current_left_margin)
            )

            for word in words:
                word_img = self._generate_word_image(word, is_heading, heading_level)

                # Line wrap
                if cursor_x + word_img.width > self.page_width - self.base_right_margin:
                    cursor_x = current_left_margin + (35 if is_bullet else 0)
                    current_y += line_spacing
                    if current_y > self.page_height - self.base_bottom_margin:
                        pages.append(current_page_glyphs)
                        current_page_glyphs = []
                        current_y = self.base_top_margin

                delta_x = cursor_x - current_left_margin
                sag_y = int(delta_x * np.tan(line_angle_rad))
                jitter_y = random.randint(-1, 1)

                placed_x = int(cursor_x)
                placed_y = int(current_y + sag_y + jitter_y)

                # Micro-rotation
                rot_deg = random.uniform(-0.4, 0.4)
                if abs(rot_deg) > 0.15:
                    word_img = word_img.rotate(rot_deg, resample=Image.Resampling.BILINEAR, expand=True)

                current_page_glyphs.append({
                    "char": word,
                    "image": word_img,
                    "x": placed_x,
                    "y": placed_y,
                    "is_heading": is_heading
                })

                # Space after word
                cursor_x += word_img.width + random.randint(28, 38)

            # Next line
            current_y += int(line_spacing * (1.3 if is_heading else 1.0))
            if current_y > self.page_height - self.base_bottom_margin:
                pages.append(current_page_glyphs)
                current_page_glyphs = []
                current_y = self.base_top_margin

        if current_page_glyphs:
            pages.append(current_page_glyphs)

        logger.info(f"Layout quality report: {self.quality_report}")
        return pages
