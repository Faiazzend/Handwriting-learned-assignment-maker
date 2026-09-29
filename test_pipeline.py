import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

from engine.extractor import GlyphExtractor
from engine.layout import UnruledPageLayout
from engine.compositor import CamScannerCompositor
from engine.pdf_builder import AssignmentPDFBuilder

def test():
    print("[1/4] Testing GlyphExtractor...")
    extractor = GlyphExtractor()
    stats = extractor.get_stats()
    print(f"Stats: {stats['unique_characters']} characters, {stats['total_variants']} total variants")

    print("[2/4] Testing UnruledPageLayout...")
    sample_text = (
        "# Political Authority and Modern State\n\n"
        "→ Power was fragmented in early days.\n"
        "→ Authority came from management of conflict.\n"
        "→ In reality legitimacy is not abstract concept."
    )
    layout = UnruledPageLayout(extractor.index, words_index_path="data/words_v2/words_index.json")
    pages = layout.layout_document(sample_text, baseline_slant=0.4, margin_drift=10, line_spacing=95)
    print(f"Laid out {len(pages)} page(s). Total glyphs/words on Page 1: {len(pages[0])}")

    print("[3/4] Testing CamScannerCompositor...")
    compositor = CamScannerCompositor()
    rendered_page = compositor.render_page(pages[0], ink_type="Royal Blue Ballpoint", camscanner_intensity=0.85)
    print(f"Rendered image size: {rendered_page.size}")

    print("[4/4] Testing AssignmentPDFBuilder...")
    pdf_builder = AssignmentPDFBuilder()
    pdf_path = pdf_builder.create_pdf([rendered_page], filename="Test_Submission.pdf")
    print(f"Generated PDF: {pdf_path}, file size: {os.path.getsize(pdf_path)} bytes")
    print("SUCCESS: Full pipeline verified!")

if __name__ == "__main__":
    test()
