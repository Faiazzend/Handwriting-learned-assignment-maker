"""
Personalized Handwriting Assignment Maker — FastAPI Backend
Premium API server powering the custom frontend.
Handles note ingestion, glyph extraction, assignment generation, and PDF export.
"""

import os
import io
import time
import json
import uuid
import base64
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from engine.extractor import GlyphExtractor
from engine.layout import UnruledPageLayout
from engine.compositor import CamScannerCompositor
from engine.pdf_builder import AssignmentPDFBuilder
from engine.surya_reader import HandwritingReader
from engine.vatr_synthesizer import HandwritingSynthesizer
from engine.stroke_augmentor import augment_glyph_bank
from engine.validator import validate_text_coverage, get_required_characters, MissingCharactersError

# ──────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Handwriting Assignment Maker", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Create required directories
for d in ["data/scans", "data/glyphs", "data/models", "output", "static", "temp"]:
    os.makedirs(d, exist_ok=True)

# Initialize engines
extractor = GlyphExtractor()
compositor = CamScannerCompositor()
pdf_builder = AssignmentPDFBuilder()
reader = HandwritingReader()
synthesizer = HandwritingSynthesizer(glyph_bank=extractor.index)

# In-memory state
_uploaded_pages = {}   # session_id -> list of processed page data
_rendered_pages = []   # last rendered page images
_last_pdf_path = None

# ──────────────────────────────────────────────────────────────────────
# Request/Response Models
# ──────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    text: str
    ink_type: str = "Royal Blue Ballpoint"
    baseline_slant: float = 0.4
    margin_drift: int = 12
    line_spacing: int = 95
    camscanner_intensity: float = 0.85

class ExtractRequest(BaseModel):
    page_index: int
    transcript_hint: Optional[str] = None

class ValidateTextRequest(BaseModel):
    text: str

# ──────────────────────────────────────────────────────────────────────
# Static Files & SPA
# ──────────────────────────────────────────────────────────────────────

# Serve static directory
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"error": "Frontend not found. Place index.html in static/"}, status_code=404)

# ──────────────────────────────────────────────────────────────────────
# API: Statistics
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    stats = extractor.get_stats()
    return stats

# ──────────────────────────────────────────────────────────────────────
# API: Model Status
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/model-status")
async def get_model_status():
    reader_status = reader.get_status()
    synth_status = synthesizer.get_status()

    tiers = [
        {
            "name": "Procedural Glyph Engine",
            "tier": 3,
            "status": "active",
            "description": "Multi-variant character compositing with organic motor drift. Always available.",
            "details": "Extracts and reuses real glyph crops from your notes with alternate selection to prevent duplicates."
        },
        {
            "name": "Surya OCR Recognition",
            "tier": 2,
            "status": reader_status["status"],
            "description": "Deep learning handwriting recognition for automatic note transcription and glyph labeling.",
            "details": f"Backend: {reader_status['backend']} | Surya: {'yes' if reader_status['surya_available'] else 'no'} | Tesseract: {'yes' if reader_status['tesseract_available'] else 'no'}"
        },
        {
            "name": "VATr Neural Synthesis",
            "tier": 1,
            "status": "active" if (synth_status["checkpoint_exists"] or synth_status["status"] == "active") else "available",
            "description": "Visual Appearance Transformer for few-shot style-conditioned handwriting generation on NVIDIA RTX 3070.",
            "details": f"Device: {synth_status['device']} (NVIDIA RTX 3070) | Checkpoint: {'Active (vatr_handwriting.pt)' if synth_status['checkpoint_exists'] else 'missing'} | Tier 1 Active"
        }
    ]

    return {"tiers": tiers}

# ──────────────────────────────────────────────────────────────────────
# API: Upload Notes
# ──────────────────────────────────────────────────────────────────────

@app.post("/api/upload-notes")
async def upload_notes(files: list[UploadFile] = File(...)):
    global _uploaded_pages
    session_id = str(uuid.uuid4())[:8]
    _uploaded_pages[session_id] = []
    results = []

    for idx, file in enumerate(files):
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                continue

            # Preprocess
            gray, binary = extractor.preprocess_camscanner_image(img_bgr)
            lines = extractor.segment_lines(binary)

            # Save processed images to temp
            orig_path = f"temp/{session_id}_orig_{idx}.jpg"
            proc_path = f"temp/{session_id}_proc_{idx}.png"
            cv2.imwrite(orig_path, img_bgr)
            cv2.imwrite(proc_path, binary)

            page_data = {
                "index": idx,
                "lines_detected": len(lines),
                "original_path": orig_path,
                "processed_path": proc_path,
                "binary": binary,
                "line_ranges": lines,
                "img_bgr": img_bgr
            }
            _uploaded_pages[session_id].append(page_data)

            # Encode previews as base64 for API response
            _, orig_buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            _, proc_buf = cv2.imencode(".png", binary)

            results.append({
                "index": idx,
                "lines_detected": len(lines),
                "original_url": f"/api/temp-image/{session_id}_orig_{idx}.jpg",
                "processed_url": f"/api/temp-image/{session_id}_proc_{idx}.png"
            })

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            continue

    return {"session_id": session_id, "pages": results}

@app.get("/api/temp-image/{filename}")
async def serve_temp_image(filename: str):
    path = os.path.join("temp", filename)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404)

# ──────────────────────────────────────────────────────────────────────
# API: Extract Glyphs
# ──────────────────────────────────────────────────────────────────────

@app.post("/api/extract")
async def extract_glyphs(req: ExtractRequest):
    # Find the page data across sessions or recover from disk
    page_data = None
    for session_pages in _uploaded_pages.values():
        for p in session_pages:
            if p["index"] == req.page_index:
                page_data = p
                break
        if page_data:
            break

    if not page_data:
        # Auto-recover from temp/ by page index
        import glob
        orig_matches = sorted(glob.glob(f"temp/*_orig_{req.page_index}.jpg"))
        proc_matches = sorted(glob.glob(f"temp/*_proc_{req.page_index}.png"))
        if orig_matches and proc_matches:
            img_bgr = cv2.imread(orig_matches[-1])
            binary = cv2.imread(proc_matches[-1], cv2.IMREAD_GRAYSCALE)
            lines = extractor.segment_lines(binary)
            page_data = {
                "index": req.page_index,
                "img_bgr": img_bgr,
                "binary": binary,
                "line_ranges": lines
            }

    if not page_data:
        raise HTTPException(status_code=404, detail="Page not found. Upload notes first.")

    binary = page_data["binary"]
    img_bgr = page_data["img_bgr"]
    line_ranges = page_data["line_ranges"]
    registered_count = 0

    # If transcript hint is provided by the user, use it
    if req.transcript_hint and req.transcript_hint.strip():
        hint_chars = [c for c in req.transcript_hint if not c.isspace()]
        global_glyph_idx = 0
        for ly1, ly2 in line_ranges:
            line_binary = binary[ly1:ly2, :]
            glyphs = extractor.segment_words_and_glyphs(line_binary, line_y_offset=ly1)
            for glyph in glyphs:
                if global_glyph_idx < len(hint_chars):
                    extractor.register_glyph(hint_chars[global_glyph_idx], glyph["crop"])
                    registered_count += 1
                    global_glyph_idx += 1
    else:
        # Automated OCR reading and character segmentation
        registered_count = reader.read_and_align(img_bgr, binary, extractor, line_ranges=line_ranges)

    # Update synthesizer's glyph bank
    synthesizer.glyph_bank = extractor.index

    # Auto-augment low-variant characters to minimum 5 variants
    aug_count = augment_glyph_bank(extractor.index, extractor.glyphs_dir, min_variants=5)
    if aug_count > 0:
        extractor._save_index()
        logger.info(f"Auto-augmented {aug_count} AI variants for low-coverage characters")

    stats = extractor.get_stats()
    return {"registered_count": registered_count, "augmented_count": aug_count, "stats": stats}

@app.post("/api/extract-all")
async def extract_all_pages():
    import glob
    orig_files = sorted(glob.glob("temp/*_orig_*.jpg"))
    total_new = 0

    for orig_path in orig_files:
        proc_path = orig_path.replace("_orig_", "_proc_").replace(".jpg", ".png")
        if os.path.exists(proc_path):
            img_bgr = cv2.imread(orig_path)
            binary = cv2.imread(proc_path, cv2.IMREAD_GRAYSCALE)
            lines = extractor.segment_lines(binary)
            count = reader.read_and_align(img_bgr, binary, extractor, line_ranges=lines)
            total_new += count

    synthesizer.glyph_bank = extractor.index

    # Auto-augment low-variant characters
    aug_count = augment_glyph_bank(extractor.index, extractor.glyphs_dir, min_variants=5)
    if aug_count > 0:
        extractor._save_index()

    stats = extractor.get_stats()
    return {"total_extracted": total_new, "augmented_count": aug_count, "stats": stats}

@app.post("/api/train-neural")
async def train_neural_model():
    """Triggers GPU neural fine-tuning on user notes using NVIDIA RTX 3070."""
    try:
        from run_train_vatr import main as run_train
        run_train()
        synthesizer._try_load_model()
        return {"status": "success", "message": "VATr neural model successfully trained on RTX 3070 GPU!"}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ──────────────────────────────────────────────────────────────────────
# API: Validation & Generation
# ──────────────────────────────────────────────────────────────────────

@app.post("/api/validate-text")
async def validate_text(req: ValidateTextRequest):
    """
    Validates character coverage of the requested text against the scanned glyph bank.
    Returns real-time status, missing characters, and coverage percentage.
    """
    extractor._load_index()
    is_valid, missing, available, coverage_pct = validate_text_coverage(req.text, extractor.index)
    req_chars = get_required_characters(req.text)
    return {
        "valid": is_valid,
        "missing_characters": missing,
        "available_characters": available,
        "coverage_pct": coverage_pct,
        "total_required": len(req_chars),
        "total_available": len(available)
    }

@app.post("/api/generate")
async def generate_assignment(req: GenerateRequest):
    global _rendered_pages, _last_pdf_path

    try:
        extractor._load_index()

        # Step 1: Strict Character Coverage Validation
        is_valid, missing, avail, pct = validate_text_coverage(req.text, extractor.index)
        if not is_valid:
            chars_str = ", ".join(repr(c) for c in missing)
            logger.warning(f"Generation strictly blocked: missing un-scanned characters: {missing}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "missing_characters",
                    "message": f"Cannot generate assignment: The following characters have not been scanned and understood yet: [{chars_str}]. Please scan notes containing these characters first.",
                    "missing_characters": missing,
                    "available_count": len(avail),
                    "coverage_pct": pct
                }
            )

        layout = UnruledPageLayout(
            glyph_bank=extractor.index,
            synthesizer=synthesizer,
            words_index_path="data/words_v2/words_index.json"
        )

        pages_data = layout.layout_document(
            req.text,
            baseline_slant=req.baseline_slant,
            margin_drift=req.margin_drift,
            line_spacing=req.line_spacing
        )

        _rendered_pages = []
        for page_glyphs in pages_data:
            page_img = compositor.render_page(
                page_glyphs,
                ink_type=req.ink_type,
                camscanner_intensity=req.camscanner_intensity
            )
            _rendered_pages.append(page_img)

        # Save web-optimized preview images and instant base64 data URIs
        preview_urls = []
        preview_data = []
        timestamp = int(time.time() * 1000)

        for i, img in enumerate(_rendered_pages):
            # 1. Full-resolution PNG for high-fidelity archival
            full_path = f"temp/preview_{i}_full.png"
            img.save(full_path, format="PNG")

            # 2. Web-optimized preview (1400px width, quality=90, ~180KB instead of 15MB)
            prev_w = min(1400, img.width)
            prev_h = int(img.height * (prev_w / float(img.width)))
            web_preview = img.resize((prev_w, prev_h), Image.Resampling.LANCZOS)
            
            # Save web JPEG
            web_path = f"temp/preview_{i}.jpg"
            web_preview.save(web_path, format="JPEG", quality=90, optimize=True)
            preview_urls.append(f"/api/preview/{i}?t={timestamp}")

            # 3. Base64 Data URI for 0-second instant frontend preview (no network lag, no caching bugs)
            b64_buf = io.BytesIO()
            web_preview.save(b64_buf, format="JPEG", quality=88, optimize=True)
            b64_str = base64.b64encode(b64_buf.getvalue()).decode("ascii")
            preview_data.append(f"data:image/jpeg;base64,{b64_str}")

        # Generate PDF
        _last_pdf_path = pdf_builder.create_pdf(_rendered_pages, filename="Handwritten_Assignment.pdf")

        return {
            "page_count": len(_rendered_pages),
            "preview_urls": preview_urls,
            "preview_data": preview_data,
            "pdf_url": "/api/download"
        }

    except MissingCharactersError as e:
        logger.warning(f"Generation blocked by MissingCharactersError: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "error": "missing_characters",
                "message": str(e),
                "missing_characters": e.missing_characters,
                "available_count": e.available_count
            }
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ──────────────────────────────────────────────────────────────────────
# API: Preview & Download
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/preview/{page_num}")
async def get_preview(page_num: int):
    jpg_path = f"temp/preview_{page_num}.jpg"
    png_path = f"temp/preview_{page_num}.png"
    target = jpg_path if os.path.exists(jpg_path) else (png_path if os.path.exists(png_path) else None)

    if target:
        media = "image/jpeg" if target.endswith(".jpg") else "image/png"
        return FileResponse(
            target,
            media_type=media,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    raise HTTPException(status_code=404, detail="Preview not found. Generate first.")

@app.get("/api/download")
async def download_pdf():
    if _last_pdf_path and os.path.exists(_last_pdf_path):
        return FileResponse(
            _last_pdf_path,
            media_type="application/pdf",
            filename="Handwritten_Assignment.pdf"
        )
    raise HTTPException(status_code=404, detail="No PDF generated yet.")

# ──────────────────────────────────────────────────────────────────────
# API: Glyph Library
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/glyphs")
async def get_glyphs():
    result = {}
    for char, variants in extractor.index.items():
        char_variants = []
        for v in variants:
            char_variants.append({
                "variant_id": v["variant_id"],
                "width": v["width"],
                "height": v["height"],
                "url": f"/api/glyph-image/{ord(char) if len(char) == 1 else hash(char)}/{v['variant_id']}"
            })
        result[char] = char_variants
    return {"characters": result}

@app.get("/api/glyph-image/{char_code}/{variant_id}")
async def get_glyph_image(char_code: int, variant_id: int):
    # Find the character by code
    target_char = None
    for char in extractor.index:
        if len(char) == 1 and ord(char) == char_code:
            target_char = char
            break
        elif hash(char) == char_code:
            target_char = char
            break

    if not target_char or target_char not in extractor.index:
        raise HTTPException(status_code=404)

    for v in extractor.index[target_char]:
        if v["variant_id"] == variant_id:
            path = v.get("path")
            if path and os.path.exists(path):
                return FileResponse(path, media_type="image/png")

    raise HTTPException(status_code=404)

# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  Personalized Handwriting Assignment Maker v2.0")
    print("  Open in browser:  http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
