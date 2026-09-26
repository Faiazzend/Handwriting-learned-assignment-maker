"""
Layout Engine: Text to Unruled Plain A4 Page Layout
Simulates natural human motor variance on blank paper with smart typographical adjustment.

Features:
- Strict Scanned Character Enforcement: Zero synthetic computer fallback fonts.
- Smart Typographical Baseline Engine: Standardized x-height, ascenders, descenders, and punctuation.
- Smart Automatic Scaling & Stroke Normalization: Uniform pen thickness and natural proportions.
- Contour-Aware Kerning: Natural character proximity and authentic ligature flow.
- Organic Motor Drift: Subtle baseline angle, margin wavering, and micro-jitter.
"""

import os
import json
import random
import cv2
import numpy as np
from PIL import Image

from engine.validator import validate_text_coverage, MissingCharactersError

# Typographical character classifications
X_HEIGHT_CHARS = set("acemnorsuvwxz")
ASCENDER_CHARS = set("bdfhkltABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?/()[]{}&→")
DESCENDER_CHARS = set("gjpqy")
PUNCTUATION_BASELINE = set(".,…")
PUNCTUATION_MID = set("-:;=")
PUNCTUATION_HIGH = set("'\"^")


class SmartGlyphAdjuster:
    """
    Normalizes glyph size, stroke thickness, and determines precise typographical
    baseline offsets for natural handwriting synthesis.
    """
    def __init__(self, target_line_height=95):
        self.line_h = target_line_height
        self.x_height = 28
        self.ascender_height = 56
        self.descender_depth = 24
        self.baseline_y = int(target_line_height * 0.68)  # ~65px from line top

    def normalize_stroke(self, img_rgba):
        """
        Normalizes ink stroke thickness and cleans pixel noise.
        """
        arr = np.array(img_rgba)
        alpha = arr[:, :, 3]
        if np.count_nonzero(alpha) == 0:
            return img_rgba

        _, bin_mask = cv2.threshold(alpha, 50, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        smoothed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

        # Soft antialiasing on edges
        smoothed_f = cv2.GaussianBlur(smoothed.astype(np.float32), (3, 3), 0.5)
        arr[:, :, 3] = np.clip(smoothed_f, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def adjust_glyph(self, char, raw_img, scale_factor=1.0):
        """
        Smartly adjusts glyph size and determines its typographical offset from baseline.
        Returns: (adjusted_image, y_offset_from_baseline)
        """
        # Trim transparent margins
        bbox = raw_img.getbbox()
        if bbox:
            img = raw_img.crop(bbox)
        else:
            img = raw_img

        w, h = img.size
        if w == 0 or h == 0:
            return img, 0

        # Determine target height based on character's typographical role
        if char in X_HEIGHT_CHARS:
            target_h = int(self.x_height * scale_factor)
            scale = target_h / float(h)
            new_w = max(int(w * scale), 6)
            new_h = target_h
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            y_offset = -new_h  # sits on baseline

        elif char in ASCENDER_CHARS:
            target_h = int(self.ascender_height * scale_factor)
            if char in "t":  # t is slightly shorter than l/k/h
                target_h = int(target_h * 0.84)
            elif char in "0123456789":
                target_h = int(target_h * 0.90)
            elif char == "→":
                target_h = int(self.x_height * 0.85 * scale_factor)

            scale = target_h / float(h)
            new_w = max(int(w * scale), 6)
            new_h = target_h
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            if char == "→":
                y_offset = -int(self.x_height * scale_factor * 0.5) - int(new_h * 0.5)
            else:
                y_offset = -new_h  # sits on baseline

        elif char in DESCENDER_CHARS:
            # Full height covers x_height + descender_depth
            target_h = int((self.x_height + self.descender_depth) * scale_factor)
            scale = target_h / float(h)
            new_w = max(int(w * scale), 6)
            new_h = target_h
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            # Body sits on baseline, tail extends below
            y_offset = -int(self.x_height * scale_factor)

        elif char in PUNCTUATION_BASELINE:
            target_h = int((8 if char == "." else 14) * scale_factor)
            scale = target_h / float(h)
            new_w = max(int(w * scale), 4)
            new_h = target_h
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            y_offset = -new_h if char == "." else -int(new_h * 0.6)

        elif char in PUNCTUATION_MID:
            target_h = int(8 * scale_factor)
            scale = target_h / float(h)
            new_w = max(int(w * scale), 8)
            new_h = target_h
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            y_offset = -int(self.x_height * scale_factor * 0.5)

        elif char in PUNCTUATION_HIGH:
            target_h = int(12 * scale_factor)
            scale = target_h / float(h)
            new_w = max(int(w * scale), 4)
            new_h = target_h
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            y_offset = -int(self.ascender_height * scale_factor) + 4

        else:
            target_h = int(self.x_height * scale_factor)
            scale = target_h / float(h)
            new_w = max(int(w * scale), 6)
            new_h = target_h
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            y_offset = -new_h

        normalized = self.normalize_stroke(resized)
        return normalized, y_offset


class UnruledPageLayout:
    def __init__(self, glyph_bank=None, synthesizer=None, words_index_path=None):
        self.glyph_bank = glyph_bank or {}
        self.synthesizer = synthesizer

        # Load authentic word bank if explicitly provided
        self.word_bank = {}
        if words_index_path and os.path.exists(words_index_path):
            try:
                with open(words_index_path, "r", encoding="utf-8") as f:
                    words_list = json.load(f)
                    for item in words_list:
                        lbl = item["label"].strip().lower()
                        if lbl not in self.word_bank:
                            self.word_bank[lbl] = []
                        self.word_bank[lbl].append(item)
            except Exception:
                pass

        # A4 at 300 DPI
        self.page_width = 2480
        self.page_height = 3508

        # Margins & spacing
        self.base_left_margin = 220
        self.base_right_margin = 220
        self.base_top_margin = 260
        self.base_bottom_margin = 260
        self.base_line_height = 95

        self.adjuster = SmartGlyphAdjuster(self.base_line_height)

    def _get_authentic_word_image(self, word):
        """Looks for an exact handwritten match in user's extracted notes corpus."""
        key = word.strip().lower()
        stripped_key = key.strip(".,;:!?\"'()[]{}")

        matches = self.word_bank.get(key, []) or self.word_bank.get(stripped_key, [])
        if matches:
            chosen = random.choice(matches)
            path = chosen.get("path")
            if path and os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGBA")
                    # Sanity check: must have valid aspect ratio and dimensions
                    if img.width >= 10 and img.height >= 12:
                        return img
                except Exception:
                    pass
        return None

    def _get_glyph_for_char(self, char, last_variant_id=None):
        """
        Retrieves an authentic scanned glyph variant.
        STRICT: Never creates synthetic fallbacks. Raises MissingCharactersError if absent.
        """
        variants = self.glyph_bank.get(char, [])
        if not variants:
            raise MissingCharactersError(missing_characters=[char], available_count=len(self.glyph_bank))

        if len(variants) > 1 and last_variant_id is not None:
            pool = [v for v in variants if v.get("variant_id") != last_variant_id]
            selected = random.choice(pool) if pool else random.choice(variants)
        else:
            selected = random.choice(variants)

        path = selected.get("path")
        if not path or not os.path.exists(path):
            # Fallback to any valid variant path in the list
            for v in variants:
                if v.get("path") and os.path.exists(v["path"]):
                    selected = v
                    path = v["path"]
                    break

        if not path or not os.path.exists(path):
            raise MissingCharactersError(missing_characters=[char], available_count=len(self.glyph_bank))

        img = Image.open(path).convert("RGBA")
        return img, selected.get("variant_id")

    def _generate_word_image(self, word, is_heading=False, heading_level=0):
        """
        Generates an authentic image of a word with smart typographical adjustment.
        Prioritizes authentic whole-word crops from user notes, with smart character fallback.
        """
        scale_factor = 1.0
        if is_heading:
            scale_factor = 1.25 if heading_level == 1 else (1.15 if heading_level == 2 else 1.05)

        total_h = int(self.base_line_height * (1.3 if is_heading else 1.0))
        base_y = int(self.adjuster.baseline_y * (1.3 if is_heading else 1.0))

        # 1. Try authentic whole-word crop from user notes
        clean_key = word.strip().lower().strip(".,;:!?\"'()[]{}")
        if clean_key:
            auth_img = self._get_authentic_word_image(clean_key)
            if auth_img:
                w_orig, h_orig = auth_img.size
                ar = w_orig / float(max(h_orig, 1))
                # Validate that crop is a realistic word crop:
                # Minimum height 14px, max 55px in scan; aspect ratio roughly matches word length
                expected_ar = max(0.5, len(clean_key) * 0.40)
                if 14 <= h_orig <= 55 and (0.35 * expected_ar <= ar <= 2.5 * expected_ar):
                    # Compute anatomical target height
                    has_desc = any(c in DESCENDER_CHARS for c in clean_key)
                    has_asc = any(c in ASCENDER_CHARS for c in clean_key)

                    if has_desc and has_asc:
                        target_h = int(68 * scale_factor)
                        y_off = -int(48 * scale_factor)
                    elif has_desc:
                        target_h = int(48 * scale_factor)
                        y_off = -int(26 * scale_factor)
                    elif has_asc:
                        target_h = int(48 * scale_factor)
                        y_off = -int(48 * scale_factor)
                    else:
                        target_h = int(26 * scale_factor)
                        y_off = -int(26 * scale_factor)

                    scale = target_h / float(h_orig)
                    scaled_w = max(int(w_orig * scale), 10)
                    scaled_h = target_h
                    resized_word = auth_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

                    # Check for trailing punctuation (e.g. "state." or "struggle,")
                    trailing_punct = [c for c in word if c in ".,;:!?"]
                    if trailing_punct and trailing_punct[-1] in self.glyph_bank:
                        p_char = trailing_punct[-1]
                        p_raw, _ = self._get_glyph_for_char(p_char)
                        p_adj, p_y = self.adjuster.adjust_glyph(p_char, p_raw, scale_factor=scale_factor)

                        comp_w = scaled_w + p_adj.width + 6
                        comp = Image.new("RGBA", (comp_w, total_h), (0, 0, 0, 0))
                        dest_y = max(0, min(total_h - scaled_h, base_y + y_off))
                        comp.alpha_composite(resized_word, dest=(0, dest_y))
                        comp.alpha_composite(p_adj, dest=(scaled_w + 2, max(0, min(total_h - p_adj.height, base_y + p_y))))
                        return comp
                    else:
                        comp = Image.new("RGBA", (scaled_w + 4, total_h), (0, 0, 0, 0))
                        dest_y = max(0, min(total_h - scaled_h, base_y + y_off))
                        comp.alpha_composite(resized_word, dest=(0, dest_y))
                        return comp

        # 2. Smart typographical character assembly fallback
        char_items = []
        last_var = None

        for ch in word:
            raw_gi, vid = self._get_glyph_for_char(ch, last_var)
            last_var = vid

            adj_img, y_off = self.adjuster.adjust_glyph(ch, raw_gi, scale_factor=scale_factor)
            char_items.append((ch, adj_img, y_off))

        if not char_items:
            return Image.new("RGBA", (20, self.base_line_height), (0, 0, 0, 0))

        # Calculate word canvas size
        total_w = sum(img.width for _, img, _ in char_items) + len(char_items) * 4 + 30
        word_canvas = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
        x_curr = 4

        for ch, img, y_off in char_items:
            # Organic baseline jitter: +- 1px
            jitter_y = random.choice([-1, 0, 1])
            dest_y = base_y + y_off + jitter_y

            # Clamp destination y
            dest_y = max(0, min(total_h - img.height, dest_y))
            word_canvas.alpha_composite(img, dest=(x_curr, dest_y))

            # Smart kerning: snug fit for cursive flow
            kern = 1 if ch in "ijlt" else (2 if ch in "mwn" else random.choice([0, 1]))
            x_curr += img.width + kern

        # Crop tightly to actual content
        bbox = word_canvas.getbbox()
        if bbox:
            return word_canvas.crop((bbox[0], 0, bbox[2] + 4, total_h))
        return word_canvas

    def layout_document(self, text, baseline_slant=0.4, margin_drift=12, line_spacing=95):
        """
        Strictly validates character coverage, parses text into lines and words,
        and places them onto A4 pages with authentic motor realism.

        Raises: MissingCharactersError if any character has not been scanned.
        Returns: list of pages, each containing placed word/glyph dicts.
        """
        # Step 1: Strict Character Coverage Validation
        is_valid, missing, avail, pct = validate_text_coverage(text, self.glyph_bank)
        if not is_valid:
            raise MissingCharactersError(missing_characters=missing, available_count=len(avail))

        # Update line spacing in adjuster
        self.adjuster.line_h = line_spacing
        self.adjuster.baseline_y = int(line_spacing * 0.68)

        pages = []
        current_page_glyphs = []
        current_y = self.base_top_margin
        current_left_margin = self.base_left_margin + random.uniform(-6, 6)

        lines = text.split("\n")

        for raw_line in lines:
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
            if line.startswith("→ ") or line.startswith("-> "):
                is_bullet = True
                line = "→ " + (line[2:] if line.startswith("→ ") else line[3:])
            elif line.startswith("- ") or line.startswith("* "):
                is_bullet = True
                line = "→ " + line[2:]

            words = line.split(" ")
            words = [w for w in words if w]

            # Line baseline angle (natural motor drift on unruled paper)
            line_angle_rad = np.radians(baseline_slant + random.uniform(-0.2, 0.2))

            cursor_x = current_left_margin + (35 if is_bullet else 0)

            # Waver margin organically
            current_left_margin += random.uniform(-margin_drift * 0.1, margin_drift * 0.1)
            current_left_margin = max(self.base_left_margin - 20, min(self.base_left_margin + 30, current_left_margin))

            for word in words:
                word_img = self._generate_word_image(word, is_heading, heading_level)

                # Check line wrap
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

                # Micro-rotation for natural hand drift
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
                space_w = random.randint(28, 36)
                cursor_x += word_img.width + space_w

            # Advance to next line
            current_y += int(line_spacing * (1.3 if is_heading else 1.0))
            if current_y > self.page_height - self.base_bottom_margin:
                pages.append(current_page_glyphs)
                current_page_glyphs = []
                current_y = self.base_top_margin

        if current_page_glyphs:
            pages.append(current_page_glyphs)

        return pages
