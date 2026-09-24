"""
Layout Engine: Text to Unruled Plain A4 Page Layout
Simulates natural human motor variance on blank paper.

Features:
- Authentic Word Bank Matching: Uses real handwritten word crops directly from user notes
- Neural VATr Word Synthesis: Generates new words using trained style embeddings
- Typographical Baseline Engine: Proper ascender, x-height, and descender alignment
- Unruled paper motor drift: subtle baseline sag, margin drift, and kerning jitter
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Typographical character classifications
DESCENDERS = set("gjpqy,;")
ASCENDERS = set("bdfhkltABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?/()[]{}&→")
PUNCTUATION_BASELINE = set(".,…")


class UnruledPageLayout:
    def __init__(self, glyph_bank=None, synthesizer=None, words_index_path="data/words/words_index.json"):
        self.glyph_bank = glyph_bank or {}
        self.synthesizer = synthesizer

        # Load authentic word bank if available
        self.word_bank = {}
        if os.path.exists(words_index_path):
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

        self.fallback_font = None
        self._init_fallback_font()

    def _init_fallback_font(self):
        for path in [
            "C:/Windows/Fonts/segoepr.ttf",
            "C:/Windows/Fonts/comic.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ]:
            if os.path.exists(path):
                try:
                    self.fallback_font = ImageFont.truetype(path, size=52)
                    return
                except Exception:
                    pass
        self.fallback_font = ImageFont.load_default()

    def _get_authentic_word_image(self, word):
        """Looks for an exact handwritten match in user's extracted notes corpus."""
        key = word.strip().lower()
        # Clean punctuation from edges
        stripped_key = key.strip(".,;:!?\"'()[]{}")
        
        matches = self.word_bank.get(key, []) or self.word_bank.get(stripped_key, [])
        if matches:
            chosen = random.choice(matches)
            path = chosen.get("path")
            if path and os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGBA")
                    return img
                except Exception:
                    pass
        return None

    def _get_glyph_for_char(self, char, last_variant_id=None):
        """Retrieves a glyph image, avoiding consecutive duplicates."""
        variants = self.glyph_bank.get(char, [])
        if variants:
            if len(variants) > 1 and last_variant_id is not None:
                pool = [v for v in variants if v.get("variant_id") != last_variant_id]
                selected = random.choice(pool) if pool else random.choice(variants)
            else:
                selected = random.choice(variants)

            path = selected.get("path")
            if path and os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGBA")
                    return img, selected.get("variant_id")
                except Exception:
                    pass

        img = self._create_synthetic_glyph(char)
        return img, None

    def _create_synthetic_glyph(self, char):
        temp = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp)
        bbox = draw.textbbox((0, 0), char, font=self.fallback_font)
        w = max(bbox[2] - bbox[0] + 8, 16)
        h = max(bbox[3] - bbox[1] + 8, 24)

        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((4 - bbox[0], 4 - bbox[1]), char, font=self.fallback_font, fill=(30, 30, 40, 255))
        return img

    def _generate_word_image(self, word, is_heading=False, heading_level=0):
        """
        Generates an image of a whole word using the best available method:
        1. Authentic word match from user's scanned notes
        2. Neural synthesis (VATr on GPU)
        3. Typographical baseline character assembly
        """
        target_h = 65
        if is_heading:
            target_h = int(65 * (1.25 if heading_level == 1 else (1.15 if heading_level == 2 else 1.05)))

        # 1. Check authentic word bank first
        auth_img = self._get_authentic_word_image(word)
        if auth_img:
            # Scale proportionally to line scale
            if auth_img.height > 0:
                scale = target_h / max(auth_img.height, 1)
                new_w = max(int(auth_img.width * scale), 10)
                new_h = max(int(auth_img.height * scale), 10)
                return auth_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 2. Try neural VATr synthesis if available
        if self.synthesizer and getattr(self.synthesizer, "tier", 3) == 1:
            try:
                neural_img = self.synthesizer.generate_word(word, target_height=target_h)
                if neural_img and neural_img.width > 5:
                    return neural_img
            except Exception:
                pass

        # 3. Typographical baseline character assembly
        char_images = []
        last_var = None
        baseline_y = int(target_h * 0.72)  # 72% down from top is standard baseline

        for ch in word:
            gi, vid = self._get_glyph_for_char(ch, last_var)
            last_var = vid

            # Scale heading glyphs
            scale = 1.0
            if is_heading:
                scale = 1.25 if heading_level == 1 else (1.15 if heading_level == 2 else 1.05)
            if scale != 1.0:
                gw = int(gi.width * scale)
                gh = int(gi.height * scale)
                gi = gi.resize((gw, gh), Image.Resampling.LANCZOS)

            # Determine typographical y-offset
            if ch in DESCENDERS:
                # Descender hangs below baseline
                y_pos = baseline_y - int(gi.height * 0.45)
            elif ch in ASCENDERS:
                # Ascender sits on baseline, reaches high
                y_pos = baseline_y - gi.height
            elif ch in PUNCTUATION_BASELINE:
                # Sits directly on baseline
                y_pos = baseline_y - gi.height
            else:
                # Normal lowercase: sits on baseline
                y_pos = baseline_y - gi.height

            char_images.append((gi, y_pos))

        if not char_images:
            return Image.new("RGBA", (20, target_h), (0, 0, 0, 0))

        # Calculate bounding box of entire composed word
        total_w = sum(ci[0].width + random.randint(1, 3) for ci in char_images) + 8
        min_y = min(y for _, y in char_images)
        max_y = max(y + img.height for img, y in char_images)
        word_h = max(max_y - min_y + 10, target_h)

        word_img = Image.new("RGBA", (total_w, word_h), (0, 0, 0, 0))
        x_cursor = 4

        for ci, y_pos in char_images:
            dest_y = y_pos - min_y
            word_img.alpha_composite(ci, dest=(x_cursor, max(0, dest_y)))
            x_cursor += ci.width + random.randint(1, 3)

        return word_img

    def layout_document(self, text, baseline_slant=0.4, margin_drift=12, line_spacing=95):
        """
        Parses text into lines, generates word images, and places them onto pages
        with organic motor drift simulating unruled paper writing.

        Returns: list of pages, each page is a list of placed glyph/word dicts
        """
        pages = []
        current_page_glyphs = []
        current_y = self.base_top_margin
        current_left_margin = self.base_left_margin + random.uniform(-8, 8)

        lines = text.split("\n")

        for raw_line in lines:
            line = raw_line.strip()

            # Blank line = paragraph gap
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
            line_angle_rad = np.radians(baseline_slant + random.uniform(-0.25, 0.25))

            cursor_x = current_left_margin + (35 if is_bullet else 0)

            # Waver margin organically
            current_left_margin += random.uniform(-margin_drift * 0.12, margin_drift * 0.12)
            current_left_margin = max(self.base_left_margin - 25, min(self.base_left_margin + 35, current_left_margin))

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

                # Micro-rotation for organic feel
                rot_deg = random.uniform(-0.6, 0.6)
                if abs(rot_deg) > 0.2:
                    word_img = word_img.rotate(rot_deg, resample=Image.Resampling.BILINEAR, expand=True)

                current_page_glyphs.append({
                    "char": word,
                    "image": word_img,
                    "x": placed_x,
                    "y": placed_y,
                    "is_heading": is_heading
                })

                # Space after word
                cursor_x += word_img.width + random.randint(22, 34)

            # Advance to next line
            current_y += int(line_spacing * (1.3 if is_heading else 1.0))
            if current_y > self.page_height - self.base_bottom_margin:
                pages.append(current_page_glyphs)
                current_page_glyphs = []
                current_y = self.base_top_margin

        if current_page_glyphs:
            pages.append(current_page_glyphs)

        return pages
