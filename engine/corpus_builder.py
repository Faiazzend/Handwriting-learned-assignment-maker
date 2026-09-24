"""
Corpus Builder: Extracts clean words and character glyphs from user's scanned notes.
Uses forced line alignment and inter-word projection gap detection to isolate intact words.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image

PAGE_TRANSCRIPTS = {
    1: [
        ["→", "Political", "authority", "and", "idea", "of", "modern", "state", "came", "from", "coercion"],
        ["→", "Early", "days", "→", "Power", "was", "fragmented"],
        ["→", "They", "gained", "advantage", "being", "organized"],
        ["→", "Established", "administrative", "networks"],
        ["→", "Gradually", "centralized"],
        ["→", "This", "consolidation", "a", "feature", "of", "modern", "authority"],
        ["→", "Obedience", "of", "people", "came", "from", "because", "authority", "could", "simply", "gain", "more"],
        ["→", "Authority", "grew", "from", "mixture", "of", "things", "&", "be", "stable"],
        ["→", "State", "came", "from", "management", "of", "violence", "conflict"],
        ["→", "Creation", "of", "bureaucratic", "systems", "kept", "population", "dependent"],
        ["→", "In", "reality", "legitimacy", "is", "not", "abstract", "concept"],
        ["It", "is", "the", "practical", "effect", "of", "a", "state", "demonstrating", "that", "no", "alternative", "center", "of", "violence", "can", "challenge", "it"],
        ["→", "Citizens", "not", "only", "obey", "because", "punishment", "but", "because", "state", "provides"]
    ],
    2: [
        ["→", "Social", "contract", "theory", "is", "the", "idea", "that", "political", "authority", "of", "state", "are", "made", "because", "people", "agree", "to", "give", "up", "freedom"],
        ["→", "In", "complete", "freedom", "people", "live", "in", "fear"],
        ["→", "Thus", "people", "collectively", "form", "a", "contract", "establish", "a", "governing", "body"],
        ["→", "Different", "people", "interpret", "this", "theory", "differently"],
        ["→", "At", "its", "core", "social", "contract", "theory", "explains", "why", "people", "accept", "political", "authority"],
        ["→", "At", "its", "core", "the", "social", "contract", "is", "about", "the", "foundation", "of", "legitimate", "political", "power"],
        ["→", "It", "explains", "that", "government", "comes", "from", "collective", "decision", "of", "the", "people", "that", "bind", "everyone"],
        ["→", "Not", "a", "document", "but", "a", "philosophical", "model", "that", "people", "accept", "that", "places", "limits", "on", "their", "freedoms"]
    ],
    3: [
        ["→", "Origin", "of", "state", "lies", "not", "in", "philosophical", "agreement", "but", "in", "historic"],
        ["→", "Early", "europe", "many", "struggled"],
        ["→", "Those", "who", "organized", "had", "the", "upper-hand"],
        ["→", "To", "sustain", "institutions", "for", "taxation"],
        ["→", "Over-time", "those", "war-making", "efforts", "turned", "into", "tools", "of", "governance"],
        ["→", "Therefore", "modern", "state", "didnt", "originate", "from", "social", "contract", "but", "in", "deliberate", "attempt", "to", "organize", "war"],
        ["→", "Nature", "of", "state", "thus", "lies", "deeply", "tied", "to", "this", "history"],
        ["→", "It", "is", "an", "organization", "that", "claims", "&", "monopolizes", "legitimate", "violence", "over", "a", "territory"],
        ["→", "Stability", "bureaucracy", "and", "citizenship", "emerged", "only", "after", "violence", "was", "contained", "&", "concentrated"],
        ["→", "State", "is", "fundamentally", "a", "product", "of", "political", "struggle"]
    ]
}


def build_corpus(scans_dir="data/scans", words_dir="data/words", glyphs_dir="data/glyphs"):
    os.makedirs(words_dir, exist_ok=True)
    os.makedirs(glyphs_dir, exist_ok=True)

    from engine.extractor import GlyphExtractor
    extractor = GlyphExtractor(glyphs_dir=glyphs_dir, index_file=os.path.join(glyphs_dir, "glyphs_index.json"))

    words_index = []
    total_words_extracted = 0

    for page_num, line_transcripts in PAGE_TRANSCRIPTS.items():
        img_path = os.path.join(scans_dir, f"notes_page_{page_num}.jpg")
        if not os.path.exists(img_path):
            continue

        img_bgr = cv2.imread(img_path)
        gray, binary = extractor.preprocess_camscanner_image(img_bgr)
        lines = extractor.segment_lines(binary)

        print(f"Page {page_num}: {len(lines)} image lines vs {len(line_transcripts)} transcript lines")

        # Map each line
        for l_idx, (ly1, ly2) in enumerate(lines):
            line_words = line_transcripts[min(l_idx, len(line_transcripts) - 1)]
            line_strip = binary[ly1:ly2, :]
            line_bgr = img_bgr[ly1:ly2, :]

            # Find word bounding boxes in line using contours
            contours, _ = cv2.findContours(line_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w >= 5 and h >= 8 and (w * h >= 40):
                    boxes.append((x, y, w, h))

            if not boxes:
                continue

            # Sort boxes left to right
            boxes.sort(key=lambda b: b[0])

            # Cluster connected components that are close together into words (gap threshold ~14px)
            word_clusters = []
            curr_cluster = [boxes[0]]

            for b in boxes[1:]:
                prev_b = curr_cluster[-1]
                prev_right = prev_b[0] + prev_b[2]
                curr_left = b[0]

                if curr_left - prev_right < 14:  # Letters in same word
                    curr_cluster.append(b)
                else:  # Space between words
                    word_clusters.append(curr_cluster)
                    curr_cluster = [b]

            if curr_cluster:
                word_clusters.append(curr_cluster)

            # Map word clusters to transcript words
            for w_idx, cluster in enumerate(word_clusters):
                # Bounding box of word
                min_x = min(b[0] for b in cluster)
                min_y = min(b[1] for b in cluster)
                max_x = max(b[0] + b[2] for b in cluster)
                max_y = max(b[1] + b[3] for b in cluster)

                w_crop = max_x - min_x
                h_crop = max_y - min_y

                if w_crop < 6 or h_crop < 8:
                    continue

                word_bin = line_strip[min_y:max_y, min_x:max_x]
                word_label = line_words[min(w_idx, len(line_words) - 1)]

                # Save clean word crop
                word_filename = f"word_p{page_num}_l{l_idx}_w{w_idx}.png"
                word_path = os.path.join(words_dir, word_filename)

                # Transparent RGBA word image
                rgba = np.zeros((h_crop, w_crop, 4), dtype=np.uint8)
                rgba[:, :, 0] = 25
                rgba[:, :, 1] = 25
                rgba[:, :, 2] = 30
                rgba[:, :, 3] = word_bin
                cv2.imwrite(word_path, rgba)

                words_index.append({
                    "label": word_label,
                    "path": word_path,
                    "width": w_crop,
                    "height": h_crop,
                    "page": page_num
                })
                total_words_extracted += 1

                # If word is single character or arrow, register directly into glyph bank
                if len(word_label) == 1 or word_label == "→":
                    extractor.register_glyph(word_label, word_bin)

    with open(os.path.join(words_dir, "words_index.json"), "w", encoding="utf-8") as f:
        json.dump(words_index, f, indent=2)

    print(f"Successfully extracted {total_words_extracted} intact handwritten words!")
    return words_index


if __name__ == "__main__":
    build_corpus()
