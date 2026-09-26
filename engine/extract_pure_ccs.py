"""
Pure Connected-Component Character Extractor
Extracts only 100% genuine, unsevered, complete character components
from ground-truth words across all 3 pages.
"""

import os
import sys
import json
import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

glyphs_dir = "data/glyphs"
index_path = os.path.join(glyphs_dir, "glyphs_index.json")
words_index_path = "data/words/words_index.json"

if not os.path.exists(words_index_path):
    print("words_index.json not found")
    sys.exit(1)

with open(words_index_path, "r", encoding="utf-8") as f:
    words_list = json.load(f)

print(f"Loaded {len(words_list)} words from words_index.json")

clean_bank = {}

def register(char, crop):
    if crop is None or crop.size == 0 or np.count_nonzero(crop) == 0:
        return
    h, w = crop.shape
    if w < 4 or h < 6 or (w * h < 24):
        return

    char_code = ord(char) if len(char) == 1 else hash(char)
    cdir = os.path.join(glyphs_dir, f"char_{char_code}")
    os.makedirs(cdir, exist_ok=True)

    if char not in clean_bank:
        clean_bank[char] = []

    vid = len(clean_bank[char]) + 1
    fn = f"{vid}.png"
    fp = os.path.join(cdir, fn)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 28
    rgba[:, :, 1] = 28
    rgba[:, :, 2] = 34
    rgba[:, :, 3] = crop
    cv2.imwrite(fp, rgba)

    clean_bank[char].append({
        "variant_id": vid,
        "path": fp,
        "width": w,
        "height": h,
        "aspect_ratio": round(w / float(h), 2)
    })

# Process all words
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

    # Standalone single character / bullet / symbol
    if len(lbl) == 1 or lbl == "→":
        register(lbl, alpha)
        continue

    # Clean characters (strip punctuation from edges if needed)
    raw_chars = [c for c in lbl if not c.isspace()]
    if not raw_chars:
        continue

    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(alpha)
    if num_labels <= 1:
        continue

    # Sort CCs left to right
    ccs = []
    for l in range(1, num_labels):
        x, y, w, h, area = stats[l, 0], stats[l, 1], stats[l, 2], stats[l, 3], stats[l, 4]
        if area >= 12 and w >= 3 and h >= 5:
            mask = np.zeros_like(alpha)
            mask[labels_im == l] = 255
            crop = mask[y:y+h, x:x+w]
            ccs.append((x, crop))

    ccs.sort(key=lambda item: item[0])

    # Case 1: Exact 1-to-1 match (every letter was written disconnected!)
    if len(ccs) == len(raw_chars):
        for ch, (_, crop) in zip(raw_chars, ccs):
            register(ch, crop)

    # Case 2: First letter is disconnected (e.g. capital letters, t, s, i)
    elif len(ccs) >= 2 and len(raw_chars) >= 2:
        x0, c0 = ccs[0]
        x1, c1 = ccs[1]
        w0 = c0.shape[1]
        if x1 >= (x0 + w0):
            register(raw_chars[0], c0)

    # Case 3: Trailing punctuation (like period or comma)
    if len(raw_chars) > 1 and raw_chars[-1] in ".,;:!?'":
        if len(ccs) >= 2:
            last_ch = raw_chars[-1]
            last_x, last_crop = ccs[-1]
            prev_x, prev_crop = ccs[-2]
            if last_x >= (prev_x + prev_crop.shape[1]) and last_crop.shape[1] <= 30 and last_crop.shape[0] <= 35:
                register(last_ch, last_crop)

    # Case 4: Hyphen in hyphenated words (e.g. war-making, Over-time)
    if "-" in raw_chars and len(ccs) >= 3:
        for x_c, crop in ccs[1:-1]:
            cw, ch = crop.shape[1], crop.shape[0]
            if 6 <= cw <= 35 and 3 <= ch <= 12 and (cw / float(max(ch, 1)) >= 1.5):
                register("-", crop)
                break

# Save index
with open(index_path, "w", encoding="utf-8") as f:
    json.dump(clean_bank, f, indent=2)

total_vars = sum(len(v) for v in clean_bank.values())
print(f"Extracted {total_vars} pure unsevered variants across {len(clean_bank)} characters:")
print(sorted(list(clean_bank.keys())))
