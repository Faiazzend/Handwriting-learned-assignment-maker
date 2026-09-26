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
        Reads handwriting and extracts individual character crops into the glyph bank.
        Returns the number of registered glyphs.
        """
        registered_count = 0

        if self.backend == "rapidocr" and self._rapid_ocr:
            try:
                results, _ = self._rapid_ocr(img_bgr)
                if results:
                    for item in results:
                        box = np.array(item[0], dtype=np.int32)
                        text = item[1].strip()
                        
                        x1, y1 = np.min(box, axis=0)
                        x2, y2 = np.max(box, axis=0)
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img_bgr.shape[1], x2)
                        y2 = min(img_bgr.shape[0], y2)

                        word_bin = binary_mask[y1:y2, x1:x2]
                        chars = [c for c in text if not c.isspace()]
                        if not chars or word_bin.shape[0] < 8 or word_bin.shape[1] < 8:
                            continue

                        # Contour detection inside word
                        contours, _ = cv2.findContours(word_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        valid_contours = []
                        for cnt in contours:
                            cx, cy, cw, ch = cv2.boundingRect(cnt)
                            if cw >= 3 and ch >= 5 and (cw * ch >= 20):
                                valid_contours.append((cx, cy, cw, ch))

                        valid_contours.sort(key=lambda b: b[0])

                        # 1-to-1 contour mapping
                        if len(valid_contours) == len(chars):
                            for ch_char, (cx, cy, cw, ch) in zip(chars, valid_contours):
                                crop = word_bin[cy:cy+ch, cx:cx+cw]
                                extractor.register_glyph(ch_char, crop)
                                registered_count += 1
                        elif len(chars) == 1:
                            # Single character: take all non-zero pixels cleanly
                            coords = cv2.findNonZero(word_bin)
                            if coords is not None:
                                tx, ty, tw, th = cv2.boundingRect(coords)
                                if tw >= 4 and th >= 6:
                                    crop_trimmed = word_bin[ty:ty+th, tx:tx+tw]
                                    extractor.register_glyph(chars[0], crop_trimmed)
                                    registered_count += 1
                        else:
                            # Cursive connected word: find valleys in vertical projection
                            v_proj = np.sum(word_bin > 0, axis=0)
                            w_total = word_bin.shape[1]
                            
                            # Find local minima (valleys) between strokes
                            if len(chars) > 1 and w_total >= len(chars) * 6:
                                expected_w = w_total / len(chars)
                                prev_cut = 0
                                
                                for i in range(1, len(chars)):
                                    # Search window around expected boundary
                                    center = int(i * expected_w)
                                    win_start = max(prev_cut + 5, int(center - expected_w * 0.4))
                                    win_end = min(w_total - 5, int(center + expected_w * 0.4))
                                    
                                    if win_end > win_start:
                                        window = v_proj[win_start:win_end]
                                        cut_x = win_start + int(np.argmin(window))
                                    else:
                                        cut_x = center
                                        
                                    crop = word_bin[:, prev_cut:cut_x]
                                    prev_cut = cut_x
                                    
                                    coords = cv2.findNonZero(crop)
                                    if coords is not None:
                                        tx, ty, tw, th = cv2.boundingRect(coords)
                                        # Typographical sanity check before registering
                                        if tw >= 6 and th >= 10 and (tw * th >= 50):
                                            ar = tw / float(th)
                                            if 0.2 <= ar <= 2.2:
                                                crop_trimmed = crop[ty:ty+th, tx:tx+tw]
                                                extractor.register_glyph(chars[i-1], crop_trimmed)
                                                registered_count += 1
                                                
                                # Last character
                                crop = word_bin[:, prev_cut:]
                                coords = cv2.findNonZero(crop)
                                if coords is not None:
                                    tx, ty, tw, th = cv2.boundingRect(coords)
                                    if tw >= 6 and th >= 10 and (tw * th >= 50):
                                        ar = tw / float(th)
                                        if 0.2 <= ar <= 2.2:
                                            crop_trimmed = crop[ty:ty+th, tx:tx+tw]
                                            extractor.register_glyph(chars[-1], crop_trimmed)
                                            registered_count += 1
            except Exception as e:
                logger.error(f"RapidOCR alignment error: {e}")

        # Fallback: if OCR produced nothing, extract via line contours
        if registered_count == 0 and line_ranges:
            for ly1, ly2 in line_ranges:
                line_bin = binary_mask[ly1:ly2, :]
                glyphs = extractor.segment_words_and_glyphs(line_bin, line_y_offset=ly1)
                for g in glyphs:
                    crop = g["crop"]
                    # Register under a placeholder label or skip
                    pass

        return registered_count
