"""
Pure Glyph Bank Builder v3: Manual-Quality Character Extraction
Uses vertical projection profiles within each line to segment individual characters,
then matches them against OCR word boxes for labeling.

Key insight: Instead of 1:1 sequential matching (which drifts), we:
1. Segment lines into WORD groups using large horizontal gaps
2. OCR each word group independently for more accurate text
3. Match characters within each word group
"""

import os
import sys
import json
import shutil
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.glyph_quality import is_valid_single_glyph

SCANS_DIR = "data/scans"
OUTPUT_DIR = "data/glyphs_v2"
INDEX_FILE = "data/glyphs_v2/glyphs_index.json"

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def preprocess_scan(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    norm = cv2.divide(gray, bg, scale=255.0).clip(0, 255).astype(np.uint8)
    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 12
    )
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return gray, cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)


def segment_lines(binary):
    proj = np.sum(binary, axis=1)
    h = binary.shape[0]
    threshold = np.mean(proj) * 0.12
    in_line, lines, start_y = False, [], 0
    for y, val in enumerate(proj):
        if not in_line and val > threshold:
            in_line, start_y = True, max(0, y - 4)
        elif in_line and val <= threshold:
            in_line = False
            end_y = min(h, y + 4)
            if end_y - start_y > 12:
                lines.append((start_y, end_y))
    if in_line and (h - start_y > 12):
        lines.append((start_y, h))
    return lines


def segment_words_in_line(line_binary, min_gap=8):
    """
    Segment a binary line image into word groups using vertical projection.
    Words are separated by gaps wider than min_gap pixels.
    Returns list of (x_start, x_end) for each word.
    """
    vproj = np.sum(line_binary, axis=0)
    w = line_binary.shape[1]
    threshold = np.mean(vproj[vproj > 0]) * 0.05 if np.any(vproj > 0) else 0
    
    in_stroke = False
    segments = []
    start_x = 0
    
    for x, val in enumerate(vproj):
        if not in_stroke and val > threshold:
            in_stroke = True
            start_x = x
        elif in_stroke and val <= threshold:
            in_stroke = False
            segments.append((start_x, x))
    if in_stroke:
        segments.append((start_x, w))
    
    # Merge segments that are very close (< min_gap) into words
    if not segments:
        return []
    
    words = [segments[0]]
    for seg in segments[1:]:
        prev_end = words[-1][1]
        gap = seg[0] - prev_end
        if gap < min_gap:
            # Merge with previous
            words[-1] = (words[-1][0], seg[1])
        else:
            words.append(seg)
    
    # Filter tiny segments (noise)
    words = [(x1, x2) for x1, x2 in words if x2 - x1 >= 5]
    return words


def extract_chars_from_word(word_binary, word_x_offset=0):
    """
    Extract individual character connected components from a word crop.
    Returns list of component dicts sorted left-to-right.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        word_binary, connectivity=8
    )
    
    components = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < 10 or w < 2 or h < 2:
            continue
        
        mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
        components.append({
            'x': x + word_x_offset, 'y': y, 'w': w, 'h': h,
            'local_x': x, 'area': area, 'crop': mask,
        })
    
    components.sort(key=lambda c: c['local_x'])
    return components


def merge_i_dots(components, line_h):
    """Merge small dots with their body strokes (for i, j, etc.)."""
    if len(components) < 2:
        return components
    
    merged = []
    skip = set()
    
    for i, comp in enumerate(components):
        if i in skip:
            continue
        is_dot = comp['area'] < 40 and comp['h'] < 10 and comp['w'] < 10
        if is_dot:
            best_j = None
            best_dist = 999
            for j, other in enumerate(components):
                if j == i or j in skip or other['h'] < 8:
                    continue
                h_dist = abs((comp['local_x'] + comp['w']/2) - (other['local_x'] + other['w']/2))
                v_gap = other['y'] - (comp['y'] + comp['h'])
                if h_dist < max(other['w'], 8) and -3 < v_gap < 12:
                    dist = h_dist + abs(v_gap)
                    if dist < best_dist:
                        best_dist, best_j = dist, j
            if best_j is not None:
                body = components[best_j]
                min_x = min(comp['local_x'], body['local_x'])
                min_y = min(comp['y'], body['y'])
                max_x = max(comp['local_x'] + comp['w'], body['local_x'] + body['w'])
                max_y = max(comp['y'] + comp['h'], body['y'] + body['h'])
                nw, nh = max_x - min_x, max_y - min_y
                mc = np.zeros((nh, nw), dtype=np.uint8)
                dy, dx = comp['y'] - min_y, comp['local_x'] - min_x
                mc[dy:dy+comp['h'], dx:dx+comp['w']] = np.maximum(
                    mc[dy:dy+comp['h'], dx:dx+comp['w']], comp['crop'])
                dy, dx = body['y'] - min_y, body['local_x'] - min_x
                mc[dy:dy+body['h'], dx:dx+body['w']] = np.maximum(
                    mc[dy:dy+body['h'], dx:dx+body['w']], body['crop'])
                skip.add(best_j)
                merged.append({
                    'x': min_x + (comp['x'] - comp['local_x']),
                    'y': min_y, 'w': nw, 'h': nh,
                    'local_x': min_x, 'area': comp['area'] + body['area'], 'crop': mc,
                })
                continue
        merged.append(comp)
    
    merged.sort(key=lambda c: c['local_x'])
    return merged


def ocr_word_crop(img_bgr_crop):
    """OCR a single word crop. Returns recognized text or empty string."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
        result, _ = ocr(img_bgr_crop)
        if result:
            texts = [r[1] for r in result]
            return "".join(texts)
    except:
        pass
    return ""


def register_glyph(index, char_label, crop_binary, source_page, source_line):
    """Save a validated glyph to disk."""
    if char_label not in index:
        index[char_label] = []
    if len(index[char_label]) >= 15:
        return False
    
    var_id = len(index[char_label]) + 1
    char_code = ord(char_label) if len(char_label) == 1 else abs(hash(char_label)) % 100000
    char_dir = os.path.join(OUTPUT_DIR, f"char_{char_code}")
    os.makedirs(char_dir, exist_ok=True)
    
    filepath = os.path.join(char_dir, f"{var_id}.png")
    h_c, w_c = crop_binary.shape
    rgba = np.zeros((h_c, w_c, 4), dtype=np.uint8)
    rgba[:, :, 0] = 30
    rgba[:, :, 1] = 30
    rgba[:, :, 2] = 35
    rgba[:, :, 3] = crop_binary
    cv2.imwrite(filepath, rgba)
    
    index[char_label].append({
        "variant_id": var_id,
        "path": filepath.replace("\\", "/"),
        "width": w_c, "height": h_c,
        "aspect_ratio": round(w_c / max(h_c, 1), 2),
        "source_page": source_page, "source_line": source_line,
    })
    return True


def build_glyph_bank():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    index = {}
    total_reg, total_rej = 0, 0
    
    scan_files = sorted([
        f for f in os.listdir(SCANS_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"Found {len(scan_files)} scan pages")
    
    for scan_file in scan_files:
        scan_path = os.path.join(SCANS_DIR, scan_file)
        img_bgr = cv2.imread(scan_path)
        if img_bgr is None:
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {scan_file} ({img_bgr.shape[1]}x{img_bgr.shape[0]})")
        
        # Skip very low-res images (the 229x320 page 4 is too small for reliable extraction)
        if img_bgr.shape[0] < 400 or img_bgr.shape[1] < 400:
            print(f"  SKIP: Image too small for reliable extraction")
            continue
        
        gray, binary = preprocess_scan(img_bgr)
        lines = segment_lines(binary)
        print(f"  {len(lines)} lines detected")
        
        page_reg, page_rej = 0, 0
        
        for li, (ly1, ly2) in enumerate(lines):
            line_binary = binary[ly1:ly2, :]
            line_h = ly2 - ly1
            
            # Segment into word groups by large gaps
            word_ranges = segment_words_in_line(line_binary, min_gap=8)
            
            if not word_ranges:
                continue
            
            line_reg, line_rej = 0, 0
            
            for wx1, wx2 in word_ranges:
                word_bin = line_binary[:, wx1:wx2]
                word_w = wx2 - wx1
                
                # Skip tiny word segments
                if word_w < 8:
                    continue
                
                # Extract character components within this word
                raw_ccs = extract_chars_from_word(word_bin, word_x_offset=wx1)
                ccs = merge_i_dots(raw_ccs, line_h)
                
                if not ccs:
                    continue
                
                # OCR the word crop from the original color image
                pad = 3
                word_img = img_bgr[max(0,ly1-pad):min(img_bgr.shape[0],ly2+pad), 
                                   max(0,wx1-pad):min(img_bgr.shape[1],wx2+pad)]
                
                ocr_text = ocr_word_crop(word_img)
                
                if not ocr_text.strip():
                    continue
                
                # Clean: remove spaces from OCR text for char-level matching
                chars = [ch for ch in ocr_text if not ch.isspace()]
                
                # Only align if the count roughly matches
                # Allow some tolerance for OCR miscount
                ratio = len(chars) / max(len(ccs), 1)
                if ratio < 0.4 or ratio > 2.5:
                    # Too much mismatch, skip this word
                    continue
                
                # Sequential alignment within the word
                for ci, comp in enumerate(ccs):
                    if ci >= len(chars):
                        break
                    
                    char_label = chars[ci]
                    if char_label.isspace():
                        continue
                    
                    crop = comp['crop']
                    is_valid, reason = is_valid_single_glyph(crop, char_label, strict=True)
                    
                    if not is_valid:
                        line_rej += 1
                        continue
                    
                    if register_glyph(index, char_label, crop, scan_file, li):
                        line_reg += 1
            
            if line_reg > 0:
                print(f"  L{li:2d}: {len(word_ranges):2d} words | reg={line_reg} rej={line_rej}")
            
            page_reg += line_reg
            page_rej += line_rej
        
        print(f"  Page: {page_reg} registered, {page_rej} rejected")
        total_reg += page_reg
        total_rej += page_rej
    
    # Save index
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"GLYPH BANK v2 BUILD COMPLETE")
    print(f"Total registered: {total_reg}")
    print(f"Total rejected:   {total_rej}")
    print(f"Unique characters: {len(index)}")
    for ch in sorted(index.keys()):
        n = len(index[ch])
        sizes = [(v['width'], v['height']) for v in index[ch][:3]]
        print(f"  {repr(ch):6s} -> {n:2d} variants  {sizes}")
    
    return index


if __name__ == "__main__":
    build_glyph_bank()
