"""
Extractor Engine: Scanned Note Processing & Glyph Bank Builder
Handles illumination correction, background subtraction, line and word segmentation,
and contour-based character extraction.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image

class GlyphExtractor:
    def __init__(self, glyphs_dir="data/glyphs_v2", index_file="data/glyphs_v2/glyphs_index.json"):
        self.glyphs_dir = glyphs_dir
        self.index_file = index_file
        os.makedirs(self.glyphs_dir, exist_ok=True)
        self.index = self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2)

    def preprocess_camscanner_image(self, img_bgr):
        """
        Removes background unevenness, lighting gradients, and extracts clean ink mask.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Estimate background illumination using a large morphological close
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Normalize illumination: divide gray by bg
        norm = cv2.divide(gray, bg, scale=255.0)
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        
        # Adaptive thresholding to obtain crisp binary mask of ink
        binary = cv2.adaptiveThreshold(
            norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 25, 12
        )
        
        # Remove small specks of noise
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
        
        return gray, clean_binary

    def segment_lines(self, binary_img):
        """
        Segments lines of text using horizontal projection profile.
        """
        proj = np.sum(binary_img, axis=1)
        h, w = binary_img.shape
        threshold = np.mean(proj) * 0.15
        
        in_line = False
        lines = []
        start_y = 0
        
        for y, val in enumerate(proj):
            if not in_line and val > threshold:
                in_line = True
                start_y = max(0, y - 5)
            elif in_line and val <= threshold:
                in_line = False
                end_y = min(h, y + 5)
                if end_y - start_y > 15:  # filter tiny noise lines
                    lines.append((start_y, end_y))
        
        if in_line and (h - start_y > 15):
            lines.append((start_y, h))
            
        return lines

    def segment_words_and_glyphs(self, binary_line, line_y_offset=0):
        """
        Segments character blobs and words from a single text line strip.
        """
        # Find contours of connected components
        contours, hierarchy = cv2.findContours(
            binary_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        glyphs = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter noise (dust/dots)
            if w < 4 or h < 6 or (w * h < 30):
                continue
            
            crop = binary_line[y:y+h, x:x+w]
            glyphs.append({
                "bbox": (x, y + line_y_offset, w, h),
                "crop": crop,
                "aspect_ratio": round(w / float(h), 2)
            })
            
        # Sort glyphs left-to-right
        glyphs = sorted(glyphs, key=lambda g: g["bbox"][0])
        return glyphs

    def register_glyph(self, char_label, glyph_crop_binary, metadata=None):
        """
        Saves a verified glyph variation to disk with a transparent PNG background and records in index.
        Rejects noise specks, multi-char fragments, and invalid crops via strict quality gating.
        """
        if glyph_crop_binary is None or len(glyph_crop_binary.shape) < 2:
            return None

        h, w = glyph_crop_binary.shape
        if w < 2 or h < 2 or np.count_nonzero(glyph_crop_binary) == 0:
            return None

        # Quality Gate: strictly reject corrupt or non-single-glyph crops
        from engine.glyph_quality import is_valid_single_glyph
        is_v, reason = is_valid_single_glyph(glyph_crop_binary, char_label, strict=True)
        if not is_v:
            return None

        char_dir = os.path.join(self.glyphs_dir, f"char_{ord(char_label) if len(char_label) == 1 else 'bigram_' + char_label}")
        os.makedirs(char_dir, exist_ok=True)
        
        if char_label not in self.index:
            self.index[char_label] = []
            
        var_id = len(self.index[char_label]) + 1
        filename = f"{var_id}.png"
        filepath = os.path.join(char_dir, filename).replace('\\', '/')
        
        # Convert binary crop (white on black) to transparent RGBA (black ink on transparent)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = 30   # Default dark ink R
        rgba[:, :, 1] = 30   # Default dark ink G
        rgba[:, :, 2] = 35   # Default dark ink B
        rgba[:, :, 3] = glyph_crop_binary  # Alpha channel matches ink
        
        cv2.imwrite(filepath, rgba)
        
        record = {
            "variant_id": var_id,
            "path": filepath,
            "width": w,
            "height": h,
            "aspect_ratio": round(w / float(h), 2)
        }
        if metadata:
            record.update(metadata)
            
        self.index[char_label].append(record)
        self._save_index()
        return record

    def register_word(self, word_label, word_crop_binary, metadata=None):
        """
        Saves an authentic cursive whole-word crop as transparent RGBA PNG
        and records it in the words index.
        """
        if word_crop_binary is None or len(word_crop_binary.shape) < 2:
            return None
        h, w = word_crop_binary.shape
        if w < 10 or h < 8 or np.count_nonzero(word_crop_binary) == 0:
            return None

        clean_label = word_label.strip(".,;:!?'\"-")
        if not clean_label:
            return None

        words_dir = "data/words_v2"
        words_index_file = "data/words_v2/words_index.json"
        os.makedirs(words_dir, exist_ok=True)

        words_index = []
        if os.path.exists(words_index_file):
            try:
                with open(words_index_file, "r", encoding="utf-8") as f:
                    words_index = json.load(f)
            except Exception:
                words_index = []

        # Check existing variants
        existing = [w for w in words_index if w.get("label", "").lower() == clean_label.lower()]
        var_id = len(existing) + 1

        safe_name = "".join(c for c in clean_label if c.isalnum() or c in "_-")
        if not safe_name:
            safe_name = "word"
        filename = f"w_{safe_name}_{var_id}.png"
        filepath = os.path.join(words_dir, filename).replace('\\', '/')

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = 30
        rgba[:, :, 1] = 30
        rgba[:, :, 2] = 35
        rgba[:, :, 3] = word_crop_binary
        cv2.imwrite(filepath, rgba)

        record = {
            "label": clean_label,
            "path": filepath,
            "width": int(w),
            "height": int(h),
            "aspect_ratio": round(float(w) / float(h), 2)
        }
        if metadata:
            record.update(metadata)

        words_index.append(record)
        with open(words_index_file, "w", encoding="utf-8") as f:
            json.dump(words_index, f, indent=2)

        return record

    def get_stats(self):
        """Returns statistics on current glyph coverage."""
        valid_chars = {c: [v for v in vars if v.get("path") and os.path.exists(v["path"])]
                       for c, vars in self.index.items()}
        valid_chars = {c: vars for c, vars in valid_chars.items() if len(vars) > 0}
        total_chars = len(valid_chars)
        total_variants = sum(len(v) for v in valid_chars.values())
        return {
            "unique_characters": total_chars,
            "total_variants": total_variants,
            "characters": sorted(list(valid_chars.keys()))
        }

