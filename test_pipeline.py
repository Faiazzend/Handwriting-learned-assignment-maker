"""
Smoke test script to verify extractor, layout, compositor, and PDF builder end-to-end.
"""

from engine.extractor import GlyphExtractor
from engine.layout import UnruledPageLayout
from engine.compositor import CamScannerCompositor
from engine.pdf_builder import AssignmentPDFBuilder
import os

def test():
    print("[1/4] Testing GlyphExtractor...")
    extractor = GlyphExtractor()
    stats = extractor.get_stats()
    print("Stats:", stats)

    print("[2/4] Testing UnruledPageLayout...")
    sample_text = (
        "# Physics Lab Report\n"
        "Name: Alex Mercer\n"
        "Topic: Kinetic Energy and Momentum\n\n"
        "1. Theoretical Background\n"
        "Kinetic energy of an object of mass m moving with speed v is defined as:\n"
        "E_k = 0.5 * m * v^2\n\n"
        "When momentum is conserved in a closed elastic collision, both kinetic energy\n"
        "and linear momentum remain constant before and after impact."
    )
    layout = UnruledPageLayout(extractor.index)
    pages = layout.layout_document(sample_text, baseline_slant=0.4, margin_drift=10, line_spacing=95)
    print(f"Laid out {len(pages)} page(s). Total glyphs on Page 1: {len(pages[0])}")

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
