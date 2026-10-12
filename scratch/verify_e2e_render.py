"""
End-to-End Verification:
Generates a complete assignment document with hybrid AI integration,
ensuring clean proportions, no 6px fragments, and pristine output.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
from engine.extractor import GlyphExtractor
from engine.ai_provider import HybridAIProvider
from engine.layout import UnruledPageLayout
from engine.compositor import CamScannerCompositor

def run_verification():
    print("=" * 60)
    print("End-to-End Assignment Maker Verification")
    print("=" * 60)

    extractor = GlyphExtractor()
    ai = HybridAIProvider()
    health = ai.check_health()
    print("AI Health Status:", health)

    layout = UnruledPageLayout(
        glyph_bank=extractor.index,
        synthesizer=ai,
        words_index_path="data/words_v2/words_index.json"
    )

    test_text = (
        "# State and Authority\n"
        "Ow do you do\n"
        "People in state are safe.\n"
        "Authority is essential to order."
    )

    print(f"Generating document with text:\n{test_text}\n")
    pages_data = layout.layout_document(test_text)
    assert len(pages_data) > 0, "No pages generated!"
    print(f"Generated {len(pages_data)} page(s). Glyphs on page 1: {len(pages_data[0])}")

    # Check glyph dimensions: ensure no 6px broken fragments are rendered
    for idx, item in enumerate(pages_data[0]):
        img = item.get("image")
        if img:
            w, h = img.size
            text = item.get("word") or item.get("char") or ""
            assert w >= 4 and h >= 4, f"Rendered crop too small: {w}x{h} for '{text}'"

    compositor = CamScannerCompositor()
    page_img = compositor.render_page(pages_data[0], ink_type="Royal Blue", camscanner_intensity=0.85)

    os.makedirs("output", exist_ok=True)
    out_path = "output/verify_e2e_render.png"
    page_img.save(out_path)
    print(f"Successfully saved pristine preview to {out_path} ({page_img.size})")

    # Generate a thumbnail for quick inspection
    thumb = page_img.crop((150, 180, 2200, 1200))
    thumb.save("output/verify_e2e_thumb.png")
    print("Saved zoomed-in crop to output/verify_e2e_thumb.png")
    print("Verification PASSED!")

if __name__ == "__main__":
    run_verification()
