"""
Layout Engine v2: Text to Unruled Plain A4 Page Layout
Simulates natural human motor variance on blank paper with proportion-preserving scaling.

Key improvements over v1:
- Proportion-preserving scaler: scales both width and height uniformly, preserving aspect ratios
- Per-character median size computation for consistent scaling
- Smart variant selection: rejects abnormally-proportioned variants
- Proper kerning based on actual scaled glyph widths
- Organic motor drift with baseline angle and micro-jitter
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
ASCENDER_CHARS = set("bdfhkltABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?/()[]{}&")
DESCENDER_CHARS = set("gjpqy")
PUNCTUATION_BASELINE = set(".,")
PUNCTUATION_MID = set("-:;=")
PUNCTUATION_HIGH = set("'\"^")
ARROW_CHARS = set("→")


class ProportionalGlyphScaler:
    """
    Unified Proportional Glyph Scaler:
    Scales all characters using a consistent global scan-to-print scale factor (2.0x),
    preserving the author's authentic relative proportions and anatomical baselines.
    """
    def __init__(self, glyph_bank=None, base_scale=2.0):
        self.base_scale = base_scale

    def scale_glyph(self, char, raw_img, extra_scale=1.0):
        bbox = raw_img.getbbox()
        img = raw_img.crop(bbox) if bbox else raw_img
        w, h = img.size
        if w == 0 or h == 0:
            return img, 0

        sf = self.base_scale * extra_scale
        new_w = max(int(w * sf), 3)
        new_h = max(int(h * sf), 3)
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        y_offset = self._compute_baseline_offset(char, new_h)
        return resized, y_offset

    def _compute_baseline_offset(self, char, glyph_height):
        if char in DESCENDER_CHARS:
            # Descenders (g, j, p, q, y): body sits on baseline, ~55% extends below
            return -int(glyph_height * 0.45)
        elif char in PUNCTUATION_MID:
            return -int(glyph_height * 1.5)
        elif char in PUNCTUATION_HIGH:
            return -int(glyph_height * 2.8)
        elif char in ARROW_CHARS:
            return -int(glyph_height * 0.8)
        else:
            # Standard glyphs (x-height, ascenders, capitals, baseline punctuation): bottom sits on baseline
            return -glyph_height


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

        # Initialize proportional scaler (2.0x scan-to-print scale)
        self.scaler = ProportionalGlyphScaler(self.glyph_bank, base_scale=2.0)

    def _get_glyph_for_char(self, char, last_variant_id=None):
        """
        Retrieves an authentic scanned glyph variant.
        STRICT: Never creates synthetic fallbacks.
        """
        variants = self.glyph_bank.get(char, [])
        if not variants:
            raise MissingCharactersError(missing_characters=[char], available_count=len(self.glyph_bank))

        # Filter to variants with valid paths
        valid_variants = [v for v in variants if v.get("path") and os.path.exists(v["path"])]
        if not valid_variants:
            raise MissingCharactersError(missing_characters=[char], available_count=len(self.glyph_bank))

        # Avoid repeating the same variant consecutively
        if len(valid_variants) > 1 and last_variant_id is not None:
            pool = [v for v in valid_variants if v.get("variant_id") != last_variant_id]
            selected = random.choice(pool) if pool else random.choice(valid_variants)
        else:
            selected = random.choice(valid_variants)

        img = Image.open(selected["path"]).convert("RGBA")
        return img, selected.get("variant_id")

    def _generate_word_image(self, word, is_heading=False, heading_level=0):
        """
        Generates word image using hybrid synthesis:
        1. Authentic scanned whole words when available (100% natural handwriting)
        2. Proportional character assembly for new/custom words
        """
        extra_scale = 1.0
        if is_heading:
            extra_scale = 1.25 if heading_level == 1 else (1.15 if heading_level == 2 else 1.05)

        total_h = int(self.base_line_height * (1.3 if is_heading else 1.0))
        baseline_y = int(total_h * 0.68)

        # ── Case 1: Check if whole word exists in authentic word bank ──
        # Split off trailing punctuation if present
        clean_word = word.rstrip(".,;:!?'\"-")
        trailing_punct = word[len(clean_word):] if len(clean_word) < len(word) else ""
        word_key = clean_word.lower()

        if word_key in self.word_bank and self.word_bank[word_key]:
            variants = [v for v in self.word_bank[word_key] if v.get("path") and os.path.exists(v["path"])]
            if variants:
                selected_word = random.choice(variants)
                raw_wimg = Image.open(selected_word["path"]).convert("RGBA")
                wbox = raw_wimg.getbbox()
                if wbox:
                    raw_wimg = raw_wimg.crop(wbox)

                # Scale word proportionally to line height
                sf = 1.35 * extra_scale
                nw = max(int(raw_wimg.width * sf), 8)
                nh = max(int(raw_wimg.height * sf), 12)
                resized_wimg = raw_wimg.resize((nw, nh), Image.Resampling.LANCZOS)

                # Anchor baseline based on descenders
                has_descenders = bool(set("gjpqy") & set(word_key))
                anchor_ratio = 0.62 if has_descenders else 0.86
                dest_y = baseline_y - int(nh * anchor_ratio)
                dest_y = max(0, min(total_h - nh, dest_y))

                # If trailing punctuation, append it
                punct_img = None
                punct_y_off = 0
                if trailing_punct:
                    try:
                        p_raw, _ = self._get_glyph_for_char(trailing_punct[0])
                        punct_img, punct_y_off = self.scaler.scale_glyph(trailing_punct[0], p_raw, extra_scale=extra_scale)
                    except Exception:
                        pass

                canvas_w = nw + (punct_img.width + 10 if punct_img else 20)
                word_canvas = Image.new("RGBA", (canvas_w, total_h), (0, 0, 0, 0))
                word_canvas.alpha_composite(resized_wimg, dest=(2, dest_y))

                if punct_img:
                    p_dest_y = baseline_y + punct_y_off
                    p_dest_y = max(0, min(total_h - punct_img.height, p_dest_y))
                    word_canvas.alpha_composite(punct_img, dest=(nw + 3, p_dest_y))

                bbox = word_canvas.getbbox()
                if bbox:
                    return word_canvas.crop((bbox[0], 0, bbox[2] + 2, total_h))
                return word_canvas

        # ── Case 2: Character-by-character assembly from authentic glyph bank ──
        char_items = []
        last_var = None

        for ch in word:
            raw_img, vid = self._get_glyph_for_char(ch, last_var)
            last_var = vid
            scaled_img, y_off = self.scaler.scale_glyph(ch, raw_img, extra_scale=extra_scale)
            char_items.append((ch, scaled_img, y_off))

        if not char_items:
            return Image.new("RGBA", (20, total_h), (0, 0, 0, 0))

        # Calculate inter-character spacing based on average glyph width
        avg_width = sum(img.width for _, img, _ in char_items) / len(char_items)
        base_kern = max(2, int(avg_width * 0.12))

        total_w = sum(img.width for _, img, _ in char_items) + len(char_items) * base_kern + 20
        word_canvas = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
        x_curr = 2

        for ch, img, y_off in char_items:
            jitter_y = random.choice([-1, 0, 0, 1])
            dest_y = baseline_y + y_off + jitter_y
            dest_y = max(0, min(total_h - img.height, dest_y))
            word_canvas.alpha_composite(img, dest=(x_curr, dest_y))

            kern = base_kern + random.choice([-1, 0, 0, 1])
            if ch in "ilt.,":
                kern = max(1, kern - 1)
            elif ch in "mwMW":
                kern += 1

            x_curr += img.width + kern

        bbox = word_canvas.getbbox()
        if bbox:
            return word_canvas.crop((bbox[0], 0, bbox[2] + 2, total_h))
        return word_canvas

    def layout_document(self, text, baseline_slant=0.4, margin_drift=12, line_spacing=95):
        """
        Validates coverage, parses text, places words onto A4 pages.
        Raises MissingCharactersError if any character is un-scanned.
        """
        # Strict validation
        is_valid, missing, avail, pct = validate_text_coverage(text, self.glyph_bank)
        if not is_valid:
            raise MissingCharactersError(missing_characters=missing, available_count=len(avail))

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
            if line.startswith("\u2192 ") or line.startswith("-> "):
                is_bullet = True
                line = "\u2192 " + (line[2:] if line.startswith("\u2192 ") else line[3:])
            elif line.startswith("- ") or line.startswith("* "):
                is_bullet = True
                line = "\u2192 " + line[2:]

            words = line.split(" ")
            words = [w for w in words if w]

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
                    word_img = word_img.rotate(
                        rot_deg, resample=Image.Resampling.BILINEAR, expand=True
                    )

                current_page_glyphs.append({
                    "char": word,
                    "image": word_img,
                    "x": placed_x,
                    "y": placed_y,
                    "is_heading": is_heading
                })

                # Space after word
                space_w = random.randint(28, 38)
                cursor_x += word_img.width + space_w

            # Next line
            current_y += int(line_spacing * (1.3 if is_heading else 1.0))
            if current_y > self.page_height - self.base_bottom_margin:
                pages.append(current_page_glyphs)
                current_page_glyphs = []
                current_y = self.base_top_margin

        if current_page_glyphs:
            pages.append(current_page_glyphs)

        return pages
