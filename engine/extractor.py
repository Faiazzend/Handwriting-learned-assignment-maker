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
    def __init__(self, glyphs_dir="data/glyphs", index_file="data/glyphs/glyphs_index.json"):
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
        Saves a glyph variation to disk with a transparent PNG background and records in index.
        """
        char_dir = os.path.join(self.glyphs_dir, f"char_{ord(char_label) if len(char_label) == 1 else 'bigram_' + char_label}")
        os.makedirs(char_dir, exist_ok=True)
        
        if char_label not in self.index:
            self.index[char_label] = []
            
        var_id = len(self.index[char_label]) + 1
        filename = f"{var_id}.png"
        filepath = os.path.join(char_dir, filename)
        
        # Convert binary crop (white on black) to transparent RGBA (black ink on transparent)
        h, w = glyph_crop_binary.shape
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

    def get_stats(self):
        """Returns statistics on current glyph coverage."""
        total_chars = len(self.index)
        total_variants = sum(len(v) for v in self.index.values())
        return {
            "unique_characters": total_chars,
            "total_variants": total_variants,
            "characters": sorted(list(self.index.keys()))
        }
