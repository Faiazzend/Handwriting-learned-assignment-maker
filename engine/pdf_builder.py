"""
PDF Builder: Assembles high-resolution rendered pages into multi-page PDFs
Compatible with university and school submission portals.
"""

import os
import pymupdf
import io

class AssignmentPDFBuilder:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def create_pdf(self, page_images, filename="Assignment_Submission.pdf"):
        """
        Takes a list of PIL Images and saves them as a high-quality multi-page PDF.
        """
        output_path = os.path.join(self.output_dir, filename)
        doc = pymupdf.open()

        for idx, img in enumerate(page_images):
            # Save PIL image to memory buffer
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG", quality=92)
            img_data = img_bytes.getvalue()

            # Create new PDF page with standard A4 dimensions (595 x 842 points)
            page = doc.new_page(width=595, height=842)
            
            # Insert image fitting the page
            rect = pymupdf.Rect(0, 0, 595, 842)
            page.insert_image(rect, stream=img_data)

        doc.save(output_path)
        doc.close()
        return output_path
