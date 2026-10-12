"""
Vision-Guided Note Extractor:
Uses AI Provider vision models to transcribe lines and locate whole words & glyphs,
strictly rejecting broken 6-pixel slivers.
"""

import os
import json
import logging
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from engine.ai_provider import AIProvider, HybridAIProvider

logger = logging.getLogger(__name__)

PUNCTUATION_CHARS = set(".,;:!?'\"-→")


def is_valid_crop_size(width: int, height: int, text: str) -> bool:
    """Strictly validates crop dimensions, permanently banning 6px slivers."""
    clean = text.strip()
    if not clean:
        return False

    # Standalone punctuation can be small
    if len(clean) == 1 and clean in PUNCTUATION_CHARS:
        return width >= 4 and height >= 4

    # Single character: must be at least 14x14 px and reasonable aspect ratio
    if len(clean) == 1:
        if width < 14 or height < 14:
            return False
        aspect = width / max(height, 1)
        # Avoid thin vertical lines (< 0.28) or flat horizontal bars (> 2.5) for letters
        return 0.28 <= aspect <= 2.5

    # Multi-letter word: must have adequate width and height
    expected_min_w = len(clean) * 9
    if width < max(expected_min_w, 20) or height < 14:
        return False

    return True


class VisionNoteExtractor:
    """Extracts intact words and clean glyphs from notes using AI vision."""

    def __init__(self, ai_provider: Optional[AIProvider] = None, words_dir: str = "data/words_v2", glyphs_dir: str = "data/glyphs_v2"):
        self.ai = ai_provider or HybridAIProvider()
        self.words_dir = words_dir
        self.glyphs_dir = glyphs_dir
        os.makedirs(self.words_dir, exist_ok=True)
        os.makedirs(self.glyphs_dir, exist_ok=True)
        self.words_index_path = os.path.join(self.words_dir, "words_index.json")

    def process_line_image(
        self, img_bgr: np.ndarray, binary_mask: np.ndarray, line_y_range: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Processes a single line crop through AI vision and extracts valid word crops."""
        ly1, ly2 = line_y_range
        lh = ly2 - ly1
        if lh < 10:
            return []

        line_crop_bgr = img_bgr[ly1:ly2, :]
        line_binary = binary_mask[ly1:ly2, :]
        line_pil = Image.fromarray(cv2.cvtColor(line_crop_bgr, cv2.COLOR_BGR2RGB))

        raw_boxes = self.ai.transcribe_and_locate(line_pil)
        if not raw_boxes:
            return []

        valid_results = []
        page_w = img_bgr.shape[1]

        for item in raw_boxes:
            word = item.get("word", "").strip()
            box = item.get("box_2d", [])
            if not word or len(box) != 4:
                continue

            ymin, xmin, ymax, xmax = box
            # If coordinates exceed the line height, they are absolute pixel coordinates
            if ymax > lh or xmax > page_w:
                px_y1, px_x1, px_y2, px_x2 = int(ymin), int(xmin), int(ymax), int(xmax)
                if px_y1 >= ly1:
                    px_y1 -= ly1
                    px_y2 -= ly1
            elif ymax <= 1.0 and xmax <= 1.0:
                px_y1 = int(ymin * lh)
                px_y2 = int(ymax * lh)
                px_x1 = int(xmin * page_w)
                px_x2 = int(xmax * page_w)
            elif ymax <= 1000 and xmax <= 1000 and (lh > 1000 or page_w > 1000):
                px_y1 = int(ymin / 1000.0 * lh)
                px_y2 = int(ymax / 1000.0 * lh)
                px_x1 = int(xmin / 1000.0 * page_w)
                px_x2 = int(xmax / 1000.0 * page_w)
            else:
                px_y1, px_x1, px_y2, px_x2 = int(ymin), int(xmin), int(ymax), int(xmax)

            crop_w = px_x2 - px_x1
            crop_h = px_y2 - px_y1

            if not is_valid_crop_size(crop_w, crop_h, word):
                logger.debug(f"Rejecting invalid crop for '{word}': {crop_w}x{crop_h}px")
                continue

            # Ensure coordinates are within line bounds
            px_y1 = max(0, min(lh, px_y1))
            px_y2 = max(0, min(lh, px_y2))
            px_x1 = max(0, min(page_w, px_x1))
            px_x2 = max(0, min(page_w, px_x2))

            if px_x2 <= px_x1 or px_y2 <= px_y1:
                continue

            # Crop ink mask
            word_binary = line_binary[px_y1:px_y2, px_x1:px_x2]
            rgba = Image.new("RGBA", (crop_w, crop_h), (30, 30, 35, 0))
            if word_binary.size > 0:
                alpha = Image.fromarray(word_binary)
                rgba.putalpha(alpha)

            valid_results.append({
                "word": word,
                "crop": rgba,
                "width": crop_w,
                "height": crop_h,
                "bbox": [ly1 + px_y1, px_x1, ly1 + px_y2, px_x2],
                "binary": word_binary
            })

        return valid_results

    def extract_from_page(
        self, img_bgr: np.ndarray, binary_mask: np.ndarray, extractor, line_ranges: Optional[List[Tuple[int, int]]] = None
    ) -> int:
        """
        Uses Gemma 4 (12B) vision via AIProvider to read and study handwritten notes.
        Extracts whole authentic words and isolated glyphs.
        """
        if line_ranges is None:
            from engine.extractor import GlyphExtractor
            line_ranges = extractor.segment_lines(binary_mask)

        # Load existing words index
        words_data = []
        if os.path.exists(self.words_index_path):
            try:
                with open(self.words_index_path, "r", encoding="utf-8") as f:
                    words_data = json.load(f)
            except Exception:
                words_data = []

        registered_count = 0
        word_counter = len(words_data)

        for ly1, ly2 in line_ranges:
            try:
                results = self.process_line_image(img_bgr, binary_mask, (ly1, ly2))
                for item in results:
                    word = item["word"]
                    crop_img = item["crop"]
                    w, h = item["width"], item["height"]
                    wb = item["binary"]

                    # 1. Register whole word in words_v2
                    clean_label = word.rstrip(".,;:!?'\"-")
                    if clean_label:
                        word_counter += 1
                        safe_label = "".join(c for c in clean_label if c.isalnum() or c in "_-")
                        word_filename = f"w_{word_counter}_{safe_label}.png"
                        word_path = os.path.join(self.words_dir, word_filename).replace("\\", "/")
                        crop_img.save(word_path)

                        words_data.append({
                            "label": clean_label,
                            "path": word_path,
                            "width": w,
                            "height": h,
                            "page": 1,
                        })
                        registered_count += 1

                    # 2. If word is a single character or punctuation, register in glyph bank
                    if len(word) == 1 and wb is not None and wb.size > 0:
                        reg = extractor.register_glyph(word, wb)
                        if reg:
                            registered_count += 1

            except Exception as e:
                logger.warning(f"Vision extraction failed on line ({ly1}, {ly2}): {e}")

        # Save updated words index
        try:
            with open(self.words_index_path, "w", encoding="utf-8") as f:
                json.dump(words_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save words_index.json: {e}")

        return registered_count
