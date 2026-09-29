"""
AI Augmentation Runner: Augments low-variant characters and runs full pipeline test.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import os

def main():
    print("=" * 60)
    print("  AI Stroke Augmentor: Low-Variant Character Enhancement")
    print("=" * 60)

    # Load glyph index
    index_path = "data/glyphs_v2/glyphs_index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        glyph_index = json.load(f)

    # Show current variant distribution
    print("\n[1] Current variant distribution:")
    low = []
    for ch in sorted(glyph_index.keys()):
        count = len(glyph_index[ch])
        if count < 5:
            low.append((ch, count))
            print(f"  {repr(ch):6s}: {count} variants  << LOW")

    print(f"\n  {len(low)} characters have fewer than 5 variants")

    # Run AI augmentation
    print("\n[2] Running AI stroke augmentation...")
    from engine.stroke_augmentor import augment_glyph_bank

    created = augment_glyph_bank(
        glyph_index,
        glyphs_dir="data/glyphs_v2",
        min_variants=5,
        max_augments_per_source=4
    )
    print(f"  Created {created} AI-augmented variants")

    # Save updated index
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(glyph_index, f, indent=2)
    print(f"  Updated glyph index saved")

    # Show post-augmentation distribution
    print("\n[3] Post-augmentation distribution:")
    still_low = 0
    for ch in sorted(glyph_index.keys()):
        count = len(glyph_index[ch])
        marker = " << STILL LOW" if count < 5 else ""
        if count < 5:
            still_low += 1
        was_low = any(c == ch for c, _ in low)
        if was_low:
            print(f"  {repr(ch):6s}: {count} variants{marker}")

    total = sum(len(v) for v in glyph_index.values())
    print(f"\n  Total: {total} variants across {len(glyph_index)} characters")
    print(f"  Characters still below 5 variants: {still_low}")

    # Test SmartVariantSelector
    print("\n[4] Testing SmartVariantSelector N-gram awareness...")
    from engine.stroke_augmentor import SmartVariantSelector
    selector = SmartVariantSelector(history_depth=3)

    test_char = 'e'  # High-variant char
    variants = glyph_index.get(test_char, [])
    selections = []
    for _ in range(10):
        selected, vid = selector.select(test_char, variants)
        selections.append(vid)

    unique = len(set(selections))
    print(f"  10 selections of '{test_char}': {selections}")
    print(f"  Unique variants used: {unique}/{len(variants)}")

    # Check for consecutive duplicates
    consec_dupes = sum(1 for i in range(1, len(selections)) if selections[i] == selections[i-1])
    print(f"  Consecutive duplicates: {consec_dupes}")
    assert consec_dupes == 0 or len(variants) <= 1, "Consecutive duplicates detected!"
    print("  PASS: No consecutive duplicates")

    # Quick layout test
    print("\n[5] Testing full layout + compositor pipeline...")
    from engine.layout import UnruledPageLayout
    from engine.compositor import CamScannerCompositor

    layout = UnruledPageLayout(
        glyph_bank=glyph_index,
        words_index_path="data/words_v2/words_index.json"
    )

    test_text = "Political science studies power and conflict in society."
    try:
        pages = layout.layout_document(test_text)
        print(f"  Layout: {len(pages)} pages, {sum(len(p) for p in pages)} words placed")

        compositor = CamScannerCompositor()
        page_img = compositor.render_page(pages[0])
        out_path = "output/test_augmented_output.png"
        os.makedirs("output", exist_ok=True)
        page_img.save(out_path)
        print(f"  Rendered: {out_path} ({page_img.size[0]}x{page_img.size[1]})")
        print("  PASS: Full pipeline works")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("  AI Augmentation Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
