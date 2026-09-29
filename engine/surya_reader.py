"""
Handwriting Reader Engine: OCR-assisted line and character extraction.
Uses RapidOCR (pure Python ONNX, no external tesseract.exe needed) as primary engine,
with Surya and Tesseract fallbacks.
"""

import os
import logging
import cv2
import numpy as np
from PIL import Image

from engine.glyph_quality import is_valid_single_glyph

logger = logging.getLogger(__name__)

# Check backends
_RAPID_AVAILABLE = False
_SURYA_AVAILABLE = False
_TESSERACT_AVAILABLE = False

try:
    from rapidocr_onnxruntime import RapidOCR
    _RAPID_AVAILABLE = True
    logger.info("RapidOCR (ONNX) loaded successfully")
except ImportError:
    pass

try:
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    _SURYA_AVAILABLE = True
except ImportError:
    pass

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    pass


class HandwritingReader:
    """
    Reads handwritten text and aligns bounding boxes to characters.
    Priority: RapidOCR (ONNX) > Surya OCR > Tesseract
    """

    def __init__(self):
        self.backend = "none"
        self._rapid_ocr = None

        if _RAPID_AVAILABLE:
            try:
                self._rapid_ocr = RapidOCR()
                self.backend = "rapidocr"
                logger.info("HandwritingReader initialized with RapidOCR backend")
            except Exception as e:
                logger.warning(f"Failed to initialize RapidOCR: {e}")

        if self.backend == "none" and _SURYA_AVAILABLE:
            try:
                self._det_predictor = DetectionPredictor()
                self._rec_predictor = RecognitionPredictor()
                self.backend = "surya"
            except Exception:
                pass

        if self.backend == "none" and _TESSERACT_AVAILABLE:
            self.backend = "tesseract"

    def get_status(self):
        return {
            "backend": self.backend,
            "rapid_available": _RAPID_AVAILABLE,
            "surya_available": _SURYA_AVAILABLE,
            "tesseract_available": _TESSERACT_AVAILABLE,
            "status": "active" if self.backend != "none" else "unavailable"
        }

    def read_and_align(self, img_bgr, binary_mask, extractor, line_ranges=None):
        """
        Smart handwriting reader & extractor:
        1. Extracts bullet arrows (→) from margin.
        2. RapidOCR detection with confidence filtering.
        3. High-confidence cursive whole words saved intact to word bank.
        4. Validates isolated single characters with strict quality gates.
        5. Extracts disconnected leading capital letters.
        6. Extracts punctuation (. and ,).
        7. Strict validation: zero corrupt/fragmented vertical slices.
        Returns the number of registered glyphs.
        """
        registered_count = 0

        # Step 1: Scan margin for bullet arrows (→)
        if binary_mask is not None and binary_mask.shape[1] > 140:
            margin = binary_mask[:, :140]
            contours, _ = cv2.findContours(margin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                cx, cy, cw, ch = cv2.boundingRect(c)
                if 18 <= cw <= 80 and 8 <= ch <= 35 and (cw / float(ch) >= 1.25):
                    arr_crop = margin[cy:cy+ch, cx:cx+cw]
                    is_v, _ = is_valid_single_glyph(arr_crop, "→", strict=True)
                    if is_v:
                        rec = extractor.register_glyph("→", arr_crop)
                        if rec:
                            registered_count += 1

        # Step 2: OCR-assisted character and word extraction
        if self.backend == "rapidocr" and self._rapid_ocr:
            try:
                results, _ = self._rapid_ocr(img_bgr)
                if results:
                    for item in results:
                        box = np.array(item[0], dtype=np.int32)
                        text = item[1].strip()
                        score = float(item[2]) if len(item) > 2 else 0.8

                        if score < 0.50 or not text:
                            continue

                        x1, y1 = np.min(box, axis=0)
                        x2, y2 = np.max(box, axis=0)
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img_bgr.shape[1], x2)
                        y2 = min(img_bgr.shape[0], y2)

                        word_bin = binary_mask[y1:y2, x1:x2]
                        if word_bin.shape[0] < 6 or word_bin.shape[1] < 4:
                            continue

                        chars = [c for c in text if not c.isspace()]
                        if not chars:
                            continue

                        # A) Single character detection (e.g. 'a', 'I', '1')
                        if len(chars) == 1:
                            coords = cv2.findNonZero(word_bin)
                            if coords is not None:
                                tx, ty, tw, th = cv2.boundingRect(coords)
                                crop_trimmed = word_bin[ty:ty+th, tx:tx+tw]
                                is_v, _ = is_valid_single_glyph(crop_trimmed, chars[0], strict=True)
                                if is_v:
                                    rec = extractor.register_glyph(chars[0], crop_trimmed)
                                    if rec:
                                        registered_count += 1
                            continue

                        # B) High-confidence whole cursive word preservation
                        clean_word = text.strip(".,;:!?'\"-")
                        if len(clean_word) >= 2 and score >= 0.65:
                            coords = cv2.findNonZero(word_bin)
                            if coords is not None:
                                wx, wy, ww, wh = cv2.boundingRect(coords)
                                w_crop = word_bin[wy:wy+wh, wx:wx+ww]
                                if hasattr(extractor, 'register_word'):
                                    extractor.register_word(clean_word, w_crop, metadata={"score": round(score, 2)})

                        # C) Trailing punctuation extraction (. or ,)
                        if text.endswith(".") and word_bin.shape[1] > 15:
                            p_strip = word_bin[:, -15:]
                            p_coords = cv2.findNonZero(p_strip)
                            if p_coords is not None:
                                px, py, pw, ph = cv2.boundingRect(p_coords)
                                if pw <= 12 and ph <= 12:
                                    p_crop = p_strip[py:py+ph, px:px+pw]
                                    is_v, _ = is_valid_single_glyph(p_crop, ".", strict=True)
                                    if is_v:
                                        rec = extractor.register_glyph(".", p_crop)
                                        if rec:
                                            registered_count += 1

                        elif text.endswith(",") and word_bin.shape[1] > 15:
                            c_strip = word_bin[:, -15:]
                            c_coords = cv2.findNonZero(c_strip)
                            if c_coords is not None:
                                cx, cy, cw, ch_h = cv2.boundingRect(c_coords)
                                if cw <= 14 and ch_h <= 18:
                                    c_crop = c_strip[cy:cy+ch_h, cx:cx+cw]
                                    is_v, _ = is_valid_single_glyph(c_crop, ",", strict=True)
                                    if is_v:
                                        rec = extractor.register_glyph(",", c_crop)
                                        if rec:
                                            registered_count += 1

                        # D) Disconnected leading capital letter extraction
                        if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                            cap_ch = clean_word[0]
                            num_c, lbls, sts, _ = cv2.connectedComponentsWithStats(word_bin)
                            if num_c >= 3:
                                c1_w = int(sts[1, 2])
                                c1_h = int(sts[1, 3])
                                c1_gap = sts[2, 0] - (sts[1, 0] + c1_w) if num_c > 2 else 0
                                if c1_gap >= 2 and c1_h >= 12 and (c1_w / float(c1_h) <= 1.4):
                                    c1_mask = (lbls == 1).astype(np.uint8) * 255
                                    c1_crop = c1_mask[sts[1, 1]:sts[1, 1]+c1_h, sts[1, 0]:sts[1, 0]+c1_w]
                                    is_v, _ = is_valid_single_glyph(c1_crop, cap_ch, strict=True)
                                    if is_v:
                                        rec = extractor.register_glyph(cap_ch, c1_crop)
                                        if rec:
                                            registered_count += 1

                        # E) 1-to-1 contour mapping for cleanly separated characters
                        contours, _ = cv2.findContours(word_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        valid_contours = []
                        for cnt in contours:
                            cx, cy, cw, ch = cv2.boundingRect(cnt)
                            if cw >= 3 and ch >= 5 and (cw * ch >= 20):
                                valid_contours.append((cx, cy, cw, ch))

                        valid_contours.sort(key=lambda b: b[0])

                        if len(valid_contours) == len(chars):
                            for ch_char, (cx, cy, cw, ch) in zip(chars, valid_contours):
                                crop = word_bin[cy:cy+ch, cx:cx+cw]
                                is_v, _ = is_valid_single_glyph(crop, ch_char, strict=True)
                                if is_v:
                                    rec = extractor.register_glyph(ch_char, crop)
                                    if rec:
                                        registered_count += 1

            except Exception as e:
                logger.error(f"RapidOCR alignment error: {e}")

        return registered_count
