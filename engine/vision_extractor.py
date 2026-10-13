"""
Vision-Guided Note Extractor:
Features 3-Tier Execution:
1. Instant Local RapidOCR (~3s full page, 100% offline & reliable)
2. Fail-Fast Cloud Gemini (4s timeout with automatic local OCR fallback)
3. Local Ollama GPU Vision
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

    # Single character: must be at least 10x10 px and reasonable aspect ratio
    if len(clean) == 1:
        if width < 10 or height < 10:
            return False
        aspect = width / max(height, 1)
        return 0.20 <= aspect <= 3.2

    # Multi-letter word: must have adequate width and height
    expected_min_w = len(clean) * 5
    if width < max(expected_min_w, 14) or height < 10:
        return False

    return True


class VisionNoteExtractor:
    """Extracts intact words and clean glyphs from notes using AI vision and fast local OCR."""

    def __init__(self, ai_provider: Optional[AIProvider] = None, words_dir: str = "data/words_v2", glyphs_dir: str = "data/glyphs_v2"):
        self.ai = ai_provider or HybridAIProvider()
        self.words_dir = words_dir
        self.glyphs_dir = glyphs_dir
        os.makedirs(self.words_dir, exist_ok=True)
        os.makedirs(self.glyphs_dir, exist_ok=True)
        self.words_index_path = os.path.join(self.words_dir, "words_index.json")

    def process_line_image(
        self, img_bgr: np.ndarray, binary_mask: np.ndarray, line_y_range: Tuple[int, int],
        force_ocr: bool = False
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Processes a single line crop through AI vision (Gemini/Ollama) with instant RapidOCR fallback."""
        ly1, ly2 = line_y_range
        lh = ly2 - ly1
        if lh < 25:
            return [], False

        line_binary = binary_mask[ly1:ly2, :]
        if np.count_nonzero(line_binary) < 300:
            return [], False

        line_crop_bgr = img_bgr[ly1:ly2, :]
        line_pil = Image.fromarray(cv2.cvtColor(line_crop_bgr, cv2.COLOR_BGR2RGB))

        raw_boxes = []
        ai_succeeded = False
        if not force_ocr:
            try:
                raw_boxes = self.ai.transcribe_and_locate(line_pil)
                if raw_boxes:
                    ai_succeeded = True
            except Exception as e:
                logger.info(f"AI transcription error on line ({ly1}, {ly2}): {e}")

        # Fallback to local RapidOCR if AI provider timed out or returned no detections
        if not raw_boxes:
            try:
                from rapidocr_onnxruntime import RapidOCR
                ocr = RapidOCR()
                ocr_results, _ = ocr(line_crop_bgr)
                if ocr_results:
                    raw_boxes = []
                    for item in ocr_results:
                        box = np.array(item[0], dtype=np.int32)
                        text = item[1].strip()
                        x1, y1 = np.min(box, axis=0)
                        x2, y2 = np.max(box, axis=0)
                        raw_boxes.append({
                            "word": text,
                            "box_2d": [int(y1), int(x1), int(y2), int(x2)],
                            "is_pixel": True
                        })
            except Exception as e:
                logger.debug(f"RapidOCR fallback error on line ({ly1}, {ly2}): {e}")

        if not raw_boxes:
            return [], ai_succeeded

        valid_results = []
        page_w = img_bgr.shape[1]

        for item in raw_boxes:
            word = item.get("word", "").strip()
            box = item.get("box_2d", [])
            if not word or len(box) != 4:
                continue

            ymin, xmin, ymax, xmax = box
            is_pixel = item.get("is_pixel", False)

            # 1. Determine horizontal range (x1, x2) in line_crop
            if is_pixel:
                px_x1 = int(xmin)
                px_x2 = int(xmax)
            elif max(float(ymin), float(xmin), float(ymax), float(xmax)) <= 1.0:
                px_x1 = int(float(xmin) * page_w)
                px_x2 = int(float(xmax) * page_w)
            elif max(float(ymin), float(xmin), float(ymax), float(xmax)) <= 1000.0:
                px_x1 = int(float(xmin) / 1000.0 * page_w)
                px_x2 = int(float(xmax) / 1000.0 * page_w)
            else:
                px_x1 = int(xmin)
                px_x2 = int(xmax)

            # Clamp horizontal coordinates
            px_x1 = max(0, min(page_w, px_x1))
            px_x2 = max(0, min(page_w, px_x2))
            if px_x2 <= px_x1:
                continue

            # 2. Determine tight vertical range from authentic ink within [px_x1, px_x2]
            col = line_binary[:, px_x1:px_x2]
            ink_rows = np.where(np.sum(col > 0, axis=1) > 0)[0]
            if len(ink_rows) > 0:
                px_y1 = max(0, int(ink_rows[0]) - 3)
                px_y2 = min(lh, int(ink_rows[-1]) + 4)
                # Ensure minimum height of 16px around the ink center
                if px_y2 - px_y1 < 16:
                    mid = (px_y1 + px_y2) // 2
                    px_y1 = max(0, mid - 8)
                    px_y2 = min(lh, mid + 8)
            else:
                px_y1 = 0
                px_y2 = lh

            # Clamp vertical coordinates to line crop bounds
            px_y1 = max(0, min(lh, px_y1))
            px_y2 = max(0, min(lh, px_y2))

            crop_w = px_x2 - px_x1
            crop_h = px_y2 - px_y1

            if crop_w < 6 or crop_h < 6:
                continue

            if not is_valid_crop_size(crop_w, crop_h, word):
                logger.debug(f"Rejecting invalid crop for '{word}': {crop_w}x{crop_h}px")
                continue

            # Crop ink mask
            word_binary = line_binary[px_y1:px_y2, px_x1:px_x2]
            if word_binary.size == 0 or word_binary.shape[0] != crop_h or word_binary.shape[1] != crop_w:
                continue

            rgba = Image.new("RGBA", (crop_w, crop_h), (30, 30, 35, 0))
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

        return valid_results, ai_succeeded

    def extract_page_rapidocr(
        self, img_bgr: np.ndarray, binary_mask: np.ndarray, extractor,
        progress_callback: Optional[Any] = None
    ) -> int:
        """
        Instant full-page extraction using ONNX RapidOCR (~3-5s for the entire page).
        Extracts words and character glyphs directly without line segmentation delays.
        """
        if progress_callback:
            try:
                progress_callback({
                    "current_line": 1,
                    "total_lines": 1,
                    "percent": 20,
                    "words_extracted": 0,
                    "status": "processing",
                    "message": "Scanning whole page with RapidOCR ONNX engine (~3s)..."
                })
            except Exception:
                pass

        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
        results, _ = ocr(img_bgr)
        if not results:
            return 0

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
        page_h, page_w = binary_mask.shape

        if progress_callback:
            try:
                progress_callback({
                    "current_line": 1,
                    "total_lines": 1,
                    "percent": 65,
                    "words_extracted": len(results),
                    "status": "processing",
                    "message": f"Detected {len(results)} text blocks. Extracting authentic ink masks..."
                })
            except Exception:
                pass

        for item in results:
            box = np.array(item[0], dtype=np.int32)
            raw_text = item[1].strip()
            score = float(item[2]) if len(item) > 2 else 0.8
            if score < 0.40 or not raw_text:
                continue

            x1, y1 = np.min(box, axis=0)
            x2, y2 = np.max(box, axis=0)

            # Clamp bounds
            x1 = max(0, min(page_w, int(x1) - 4))
            y1 = max(0, min(page_h, int(y1) - 4))
            x2 = max(0, min(page_w, int(x2) + 4))
            y2 = max(0, min(page_h, int(y2) + 4))

            w = x2 - x1
            h = y2 - y1
            if w < 10 or h < 8:
                continue

            crop_bin = binary_mask[y1:y2, x1:x2]
            if crop_bin.size == 0 or np.count_nonzero(crop_bin) < 15:
                continue

            # Find tight ink bounds within crop
            ink_rows = np.where(np.sum(crop_bin > 0, axis=1) > 0)[0]
            ink_cols = np.where(np.sum(crop_bin > 0, axis=0) > 0)[0]
            if len(ink_rows) > 0 and len(ink_cols) > 0:
                ty1 = max(0, int(ink_rows[0]) - 2)
                ty2 = min(h, int(ink_rows[-1]) + 3)
                tx1 = max(0, int(ink_cols[0]) - 2)
                tx2 = min(w, int(ink_cols[-1]) + 3)
                crop_bin = crop_bin[ty1:ty2, tx1:tx2]
                w = tx2 - tx1
                h = ty2 - ty1

            if w < 6 or h < 6:
                continue

            # Split raw_text into separate words if multi-word block
            words_in_item = raw_text.split()
            if len(words_in_item) == 1:
                word = words_in_item[0]
                clean_label = word.rstrip(".,;:!?'\"-")
                if clean_label:
                    word_counter += 1
                    safe_label = "".join(c for c in clean_label if c.isalnum() or c in "_-")
                    word_filename = f"w_{word_counter}_{safe_label}.png"
                    word_path = os.path.join(self.words_dir, word_filename).replace("\\", "/")

                    rgba = Image.new("RGBA", (w, h), (30, 30, 35, 0))
                    alpha = Image.fromarray(crop_bin)
                    rgba.putalpha(alpha)
                    rgba.save(word_path)

                    words_data.append({
                        "label": clean_label,
                        "path": word_path,
                        "width": w,
                        "height": h,
                        "page": 1,
                    })
                    registered_count += 1

                # If single character, register in glyph bank
                if len(word) == 1:
                    reg = extractor.register_glyph(word, crop_bin)
                    if reg:
                        registered_count += 1
            else:
                # Multi-word: register full phrase as well as single characters
                for word in words_in_item:
                    clean_label = word.rstrip(".,;:!?'\"-")
                    if len(clean_label) == 1:
                        extractor.register_glyph(clean_label, crop_bin)
                        registered_count += 1

        # Save words index
        try:
            with open(self.words_index_path, "w", encoding="utf-8") as f:
                json.dump(words_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save words_index.json: {e}")

        if progress_callback:
            try:
                progress_callback({
                    "current_line": 1,
                    "total_lines": 1,
                    "percent": 100,
                    "words_extracted": registered_count,
                    "status": "completed",
                    "message": f"Successfully extracted {registered_count} authentic words & glyphs via RapidOCR!"
                })
            except Exception:
                pass

        return registered_count

    def extract_from_page(
        self, img_bgr: np.ndarray, binary_mask: np.ndarray, extractor, line_ranges: Optional[List[Tuple[int, int]]] = None,
        progress_callback: Optional[Any] = None
    ) -> int:
        """
        Executes note extraction with the currently configured backend.
        If RapidOCR mode is selected, executes instant full-page extraction (~3-5s).
        Otherwise, runs fail-fast line-by-line extraction with per-line RapidOCR fallback.
        """
        backend_name = "RapidOCR Local (Instant ~3s)"
        backend_key = "rapidocr"
        if hasattr(self.ai, "active_extraction_backend"):
            backend_key = self.ai.active_extraction_backend()
        if hasattr(self.ai, "get_extraction_display_name"):
            backend_name = self.ai.get_extraction_display_name()

        # If RapidOCR mode is active, run instant whole-page extraction (~3-5s)
        if backend_key == "rapidocr":
            return self.extract_page_rapidocr(img_bgr, binary_mask, extractor, progress_callback)

        if line_ranges is None:
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
        total_lines = len(line_ranges)

        if progress_callback:
            try:
                progress_callback({
                    "current_line": 0,
                    "total_lines": total_lines,
                    "percent": 12,
                    "words_extracted": 0,
                    "status": "processing",
                    "message": f"Starting extraction with {backend_name} ({total_lines} lines detected)..."
                })
            except Exception:
                pass

        consecutive_ai_failures = 0
        circuit_broken = False

        for line_idx, (ly1, ly2) in enumerate(line_ranges):
            pct_start = max(12, min(95, int(((line_idx + 0.1) / max(total_lines, 1)) * 85) + 12))
            active_label = "RapidOCR Fast-Track" if circuit_broken else backend_name
            if progress_callback:
                try:
                    progress_callback({
                        "current_line": line_idx + 1,
                        "total_lines": total_lines,
                        "percent": pct_start,
                        "words_extracted": registered_count,
                        "status": "processing",
                        "message": f"Reading line {line_idx + 1}/{total_lines} with {active_label} ({registered_count} words found)..."
                    })
                except Exception:
                    pass

            try:
                results, ai_used = self.process_line_image(
                    img_bgr, binary_mask, (ly1, ly2), force_ocr=circuit_broken
                )
                if not circuit_broken and backend_key in ("gemini", "ollama"):
                    if ai_used:
                        consecutive_ai_failures = 0
                    else:
                        consecutive_ai_failures += 1
                        if consecutive_ai_failures >= 2:
                            circuit_broken = True
                            logger.info(f"2 consecutive AI failures/timeouts detected. Fast-tracking remaining lines with RapidOCR.")

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
                logger.warning(f"Line ({ly1}, {ly2}) processing error: {e}")

            pct_end = min(98, int(((line_idx + 1) / max(total_lines, 1)) * 85) + 12)
            if progress_callback:
                try:
                    progress_callback({
                        "current_line": line_idx + 1,
                        "total_lines": total_lines,
                        "percent": pct_end,
                        "words_extracted": registered_count,
                        "status": "processing",
                        "message": f"Processed line {line_idx + 1} of {total_lines} ({registered_count} words extracted so far)"
                    })
                except Exception:
                    pass

        # Save updated words index
        try:
            with open(self.words_index_path, "w", encoding="utf-8") as f:
                json.dump(words_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save words_index.json: {e}")

        if progress_callback:
            try:
                progress_callback({
                    "current_line": total_lines,
                    "total_lines": total_lines,
                    "percent": 100,
                    "words_extracted": registered_count,
                    "status": "completed",
                    "message": f"Successfully extracted {registered_count} authentic words and characters!"
                })
            except Exception:
                pass

        return registered_count
