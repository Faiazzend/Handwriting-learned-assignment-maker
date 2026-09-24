# Personalized Handwriting Assignment Maker ✍️

Train directly on your real handwritten notes (scanned via CamScanner), capture your organic semi-cursive handwriting nuances, and render university/school assignments onto unruled A4 paper with full CamScanner-grade photorealism.

---

## 🚀 Quick Start

### 1. Launch the Web UI
Simply double-click:
```cmd
run_app.bat
```
Or run in terminal:
```bash
streamlit run app.py
```
This opens the interactive studio in your browser at `http://localhost:8501`.

---

## 🛠️ How It Works

### Tab 1: 📝 Assignment Studio
1. Paste or type your assignment markdown/plain text.
2. Select your pen ink style:
   - **Royal Blue Ballpoint** (classic student exam pen)
   - **Dark Gel Blue**
   - **Classic Black**
   - **Fountain Blue-Black**
3. Adjust natural motor sliders (unruled paper upward baseline angle, margin waver, line spacing, and CamScanner filter intensity).
4. Click **🚀 Render Handwritten Assignment**.
5. Inspect the live high-res page preview and click **📥 Download Multi-Page CamScanner PDF**.

### Tab 2: 📥 Train on My Notes (CamScanner)
1. Drag and drop 1 to 5 clear CamScanner scans of your actual handwritten notes.
2. The engine automatically applies **illumination normalization** to strip paper shadows and isolate crisp ink strokes.
3. Automatically detects text lines, segments words, and extracts character glyphs.
4. Click **Extract & Register Glyphs** to save them directly to your personal glyph bank.

### Tab 3: 🔤 Glyph Bank Inspector
- Inspect all learned characters, numbers, and ligatures.
- View multiple alternate variants for each letter so no two consecutive letters look identical.

---

## 📁 Project Architecture

- `engine/extractor.py`: CamScanner preprocessing, background division, line & word slicing, and glyph indexing.
- `engine/layout.py`: Unruled A4 layout engine, natural baseline slant, margin drift, dynamic kerning, and math formatting.
- `engine/compositor.py`: Physics-based ink pressure shader, paper grain, corner camera shadows, and CamScanner "Magic Color" filter.
- `engine/pdf_builder.py`: High-resolution A4 multi-page PDF generation via PyMuPDF.
- `app.py`: Streamlit local web application.
- `data/glyphs/`: Local storage for your isolated character variants.
- `output/`: Generated assignments and PDF exports.
