"""
Pure Pristine Glyph Bank Builder
Extracts only genuine, verified, anatomically validated handwritten characters
from the user's note scans. Eliminates any multi-letter collages, specks, or stretched crops.
"""

import os
import sys
import shutil
import json
import cv2
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

GLYPHS_DIR = "data/glyphs"
WORDS_INDEX_PATH = "data/words/words_index.json"
INDEX_PATH = os.path.join(GLYPHS_DIR, "glyphs_index.json")

# Clean and recreate glyphs directory
if os.path.exists(GLYPHS_DIR):
    shutil.rmtree(GLYPHS_DIR)
os.makedirs(GLYPHS_DIR, exist_ok=True)

with open(WORDS_INDEX_PATH, "r", encoding="utf-8") as f:
    words_list = json.load(f)

bank = {}

def register_glyph(char, crop_mask):
    """Saves a verified character crop and adds it to the glyph bank."""
    if crop_mask is None or crop_mask.size == 0 or np.count_nonzero(crop_mask) == 0:
        return False

    # Crop tightly to ink
    rows = np.where(np.sum(crop_mask, axis=1) > 0)[0]
    cols = np.where(np.sum(crop_mask, axis=0) > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return False
    trimmed = crop_mask[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

    h, w = trimmed.shape
    if w <= 0 or h <= 0:
        return False

    ar = w / float(h)
    area = np.count_nonzero(trimmed)

    # ──────────────────────────────────────────────────────────────────
    # Strict Typographical Validation
    # ──────────────────────────────────────────────────────────────────
    valid = False
    if char == ".":
        valid = (2 <= w <= 8 and 2 <= h <= 8 and 3 <= area <= 45)
    elif char == ",":
        valid = (2 <= w <= 12 and 4 <= h <= 18 and 0.2 <= ar <= 1.1)
    elif char == "-":
        valid = (6 <= w <= 25 and 2 <= h <= 8 and ar >= 1.3)
    elif char == "→":
        valid = (14 <= w <= 55 and 7 <= h <= 25 and 1.2 <= ar <= 3.6)
    elif char in "gpqy":
        # Descenders: hang below baseline
        valid = (3 <= w <= 30 and 10 <= h <= 48 and 0.15 <= ar <= 1.4 and area >= 20)
    elif char in "bdfhklt" or char.isupper():
        # Ascenders & Capitals: reach ascender line
        valid = (3 <= w <= 32 and 6 <= h <= 48 and 0.15 <= ar <= 1.85 and area >= 18)
    elif char in "acemnorsuvwxz":
        # Lowercase x-height body
        valid = (3 <= w <= 35 and 6 <= h <= 26 and 0.20 <= ar <= 2.2 and area >= 15)
    else:
        valid = (3 <= w <= 32 and 6 <= h <= 48 and area >= 15)

    if not valid:
        return False

    char_code = ord(char) if len(char) == 1 else hash(char)
    char_dir = os.path.join(GLYPHS_DIR, f"char_{char_code}")
    os.makedirs(char_dir, exist_ok=True)

    if char not in bank:
        bank[char] = []

    var_id = len(bank[char]) + 1
    fp = os.path.join(char_dir, f"{var_id}.png")

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 28
    rgba[:, :, 1] = 28
    rgba[:, :, 2] = 34
    rgba[:, :, 3] = trimmed
    cv2.imwrite(fp, rgba)

    bank[char].append({
        "variant_id": var_id,
        "path": fp,
        "width": w,
        "height": h,
        "aspect_ratio": round(ar, 2)
    })
    return True


print("Extracting verified character glyphs from words...")

for item in words_list:
    lbl = item["label"].strip()
    path = item["path"]
    if not os.path.exists(path):
        continue

    rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[2] < 4:
        continue

    alpha = rgba[:, :, 3]
    if np.count_nonzero(alpha) == 0:
        continue

    # Standalone symbol (e.g. arrow, single character)
    if lbl == "→" or lbl == "a":
        register_glyph(lbl, alpha)
        continue

    raw_chars = [c for c in lbl if not c.isspace()]
    if not raw_chars:
        continue

    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(alpha)
    if num_labels <= 1:
        continue

    # Filter out sub-pixel dust specks (area < 15)
    ccs = []
    for l in range(1, num_labels):
        x, y, w, h, area = stats[l, 0], stats[l, 1], stats[l, 2], stats[l, 3], stats[l, 4]
        if area >= 14 and w >= 2 and h >= 3:
            mask = np.zeros_like(alpha)
            mask[labels_im == l] = 255
            crop = mask[y:y+h, x:x+w]
            ccs.append((x, crop))

    ccs.sort(key=lambda item: item[0])

    # 1. Trailing punctuation (periods, commas)
    if len(raw_chars) > 1 and raw_chars[-1] in ".,;:":
        # Check all CCs with area >= 3 to catch small dots
        all_ccs = []
        for l in range(1, num_labels):
            x, y, w, h, area = stats[l, 0], stats[l, 1], stats[l, 2], stats[l, 3], stats[l, 4]
            if area >= 3 and w >= 2 and h >= 2:
                all_ccs.append((x, y, w, h, labels_im == l))
        all_ccs.sort(key=lambda item: item[0])
        if len(all_ccs) >= 2:
            last = all_ccs[-1]
            penult = all_ccs[-2]
            # Check if last CC is cleanly separated to the right
            if last[0] >= (penult[0] + penult[2] - 1):
                crop_mask = np.zeros_like(alpha)
                crop_mask[last[4]] = 255
                c_crop = crop_mask[last[1]:last[1]+last[3], last[0]:last[0]+last[2]]
                register_glyph(raw_chars[-1], c_crop)

    # 2. Standalone initial letters (both uppercase and lowercase with pen lift)
    if raw_chars and len(ccs) >= 2:
        x0, c0 = ccs[0]
        x1, c1 = ccs[1]
        w0 = c0.shape[1]
        if x1 >= (x0 + w0 - 2) and w0 <= 32 and c0.shape[0] <= 42:
            register_glyph(raw_chars[0], c0)

    # 3. Disconnected single letters inside word (1-to-1 match)
    if len(ccs) == len(raw_chars):
        for ch, (_, crop) in zip(raw_chars, ccs):
            register_glyph(ch, crop)

    # 4. Extract verified clean 'v' from 'government'
    if lbl == "government" and len(ccs) >= 3:
        # CC 2 is the letter 'v' (between 'o' and 'e')
        register_glyph("v", ccs[2][1])

    # 4. Hyphen in hyphenated words
    if "-" in raw_chars and len(ccs) >= 3:
        for _, crop in ccs[1:-1]:
            cw, ch = crop.shape[1], crop.shape[0]
            if 6 <= cw <= 25 and 2 <= ch <= 8:
                register_glyph("-", crop)
                break

# Also scan notes for standalone margins arrows
for page_num in [1, 2, 3]:
    scan_path = f"data/scans/page_{page_num}.jpg"
    if not os.path.exists(scan_path):
        continue
    img_bgr = cv2.imread(scan_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51)))
    norm = cv2.divide(gray, bg, scale=255.0).astype(np.uint8)
    binary = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    # Margin arrows
    margin_strip = binary[:, 20:130]
    contours, _ = cv2.findContours(margin_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 18 <= w <= 50 and 8 <= h <= 24 and (w / float(h) >= 1.3):
            arrow_crop = margin_strip[y:y+h, x:x+w]
            register_glyph("→", arrow_crop)

# Save clean index
with open(INDEX_PATH, "w", encoding="utf-8") as f:
    json.dump(bank, f, indent=2)

total_variants = sum(len(v) for v in bank.values())
print(f"Successfully built pristine glyph bank with {len(bank)} characters and {total_variants} verified variants:")
for ch, variants in sorted(bank.items()):
    dims = [(v["width"], v["height"]) for v in variants[:3]]
    print(f"  {repr(ch)}: {len(variants)} var(s) {dims}")
