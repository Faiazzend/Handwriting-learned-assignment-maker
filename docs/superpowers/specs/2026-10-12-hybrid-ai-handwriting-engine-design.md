# Hybrid AI Handwriting Engine Design Specification
**Date:** 2026-10-12  
**Status:** Approved by Human Partner  
**Target:** Personalized Handwriting Assignment Maker  

---

## 1. Executive Summary & Problem Diagnosis

### 1.1 Previous Failure Mode (Root Cause)
The assignment maker previously rendered unreadable scribbles and tiny fragmented ink blocks when generating documents (e.g. *"Ow do you do"* appearing as garbled marks). Investigation revealed three compounding root causes:
1. **Broken Crop Fragmentation:** Note extraction relied on OpenCV contour slicing and RapidOCR character projection. Instead of whole characters or words, this sliced user cursive handwriting into vertical 6×7 pixel slivers (e.g., `'w'` stored as a 6-pixel wide speck, `'O'` stored as a vertical crop).
2. **Blank Tensor Generation in Neural Stub:** The previous `VATr` synthesizer generated blank tensors (`torch.Size([1, 1, 64, 8])` filled with `1.0`s) because its training targets were flattened to uniform gray.
3. **Double Failure Fallback:** When the neural synthesizer inevitably failed its ink-quality gate, the layout engine fell back to assembling words character-by-character from those 6-pixel broken slivers, scaling them unpredictably onto the A4 canvas.

### 1.2 Target Architecture
Replace the brittle slicing and experimental stub with a **production-grade Hybrid AI Subsystem** leveraging:
- **Local Ollama with `gemma4:12b`:** Running on the user's local NVIDIA RTX 3070 GPU via `http://localhost:11434` for private, high-resolution note ingestion and offline synthesis.
- **Cloud Google Gemini API:** For rapid, low-latency render-time word synthesis when an API key is available.
- **Intelligent Fallback:** When Gemini API is available, the system uses the hybrid mix (Ollama for heavy local note extraction + Gemini for fast render-time synthesis). When Gemini API is unavailable or unconfigured, the system automatically falls back to 100% local Ollama execution on the GPU.

---

## 2. Architecture & Subsystems

```
                                  ┌────────────────────────┐
                                  │      Client (Web)      │
                                  └───────────┬────────────┘
                                              │ REST API
                                              ▼
                                  ┌────────────────────────┐
                                  │    FastAPI Backend     │
                                  └───────────┬────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         ┌─────────────────────────┐                     ┌─────────────────────────┐
         │     Note Extraction     │                     │    Document Layout      │
         │ (engine/vision_extractor│                     │   (engine/layout.py)    │
         └────────────┬────────────┘                     └────────────┬────────────┘
                      │                                               │
                      │ Calls                                         │ Calls
                      ▼                                               ▼
         ┌─────────────────────────────────────────────────────────────────────────┐
         │                     HybridAIProvider (engine/ai_provider.py)             │
         ├────────────────────────────────────┬────────────────────────────────────┤
         │  Local Ollama (RTX 3070 GPU)       │  Google Gemini API (Cloud)         │
         │  • Model: gemma4:12b               │  • Models: gemini-2.0-flash        │
         │  • Endpoint: :11434                │  • Configurable API key            │
         │  • Tasks: Heavy note vision OCR &  │  • Tasks: Fast online word         │
         │    bounding box extraction         │    synthesis at render time        │
         └────────────────────────────────────┴────────────────────────────────────┘
```

---

## 3. Detailed Component Specifications

### 3.1 `engine/ai_provider.py`
Defines the unified interface and concrete providers:

1. **`AIProvider` (Abstract Base Class):**
   - `check_health() -> Dict[str, Any]`: Verifies connectivity, active model, and latency.
   - `transcribe_and_locate(image_bytes: bytes) -> List[Dict[str, Any]]`: Takes a line or page image, prompts the vision model with structured schema, and returns word labels and bounding boxes `[ymin, xmin, ymax, xmax]`.
   - `synthesize_word(word: str, style_profile: Dict[str, Any], target_height: int) -> Optional[Image.Image]`: Produces a transparent RGBA PIL Image containing the synthesized handwritten word conforming to writer slant and stroke weight.

2. **`OllamaProvider`:**
   - Connects to `http://localhost:11434`.
   - Defaults strictly to model **`gemma4:12b`** (with fallback to `qwen3.5:4b` if specified).
   - Formats vision payloads via Ollama `/api/generate` (base64 images).
   - Generates realistic ink masks using writer style conditioning.

3. **`GeminiProvider`:**
   - Connects to Google Gemini API (`generativeai` REST interface, defaulting to `gemini-2.0-flash` / `gemini-1.5-flash`).
   - Requires API key passed via configuration or environment variable `GEMINI_API_KEY`.
   - Handles structured JSON vision responses and stroke synthesis.

4. **`HybridAIProvider`:**
   - Orchestrates the dual-pipeline:
     - Extraction calls always delegate to `OllamaProvider` (leveraging local GPU directly on high-res note scans).
     - Render-time word synthesis calls attempt `GeminiProvider` first (if configured and healthy); upon timeout or error, it falls back seamlessly to `OllamaProvider`.
     - Exposes current active state and provider statuses.

---

### 3.2 Vision-Guided Note Extraction (`engine/vision_extractor.py`)
Replaces naive OpenCV horizontal projection slicing:
1. Pre-processes scanned note images into clean contrast lines.
2. Sends line crops to the vision model with prompt requesting structured JSON:
   ```json
   [
     {"word": "Early", "box_2d": [12, 45, 95, 310]},
     {"word": "bird", "box_2d": [18, 320, 92, 540]}
   ]
   ```
3. Crops words directly out of the authentic ink mask:
   - Stores intact words in `data/words_v2/` with accurate bounds.
   - Saves initial capital letters and isolated glyphs into `data/glyphs_v2/`.
4. **Strict Quality Filters:**
   - Rejects any crop with width < 14px or height < 14px (prevents 6px broken slivers).
   - Rejects aspect ratio anomalies ($\text{width} / \text{height} < 0.2$ for multi-letter words).

---

### 3.3 Render-Time Generation in `engine/layout.py`
Word rendering in `UnruledPageLayout._generate_word_image` follows a 3-tier cascade:
1. **Case 1 (Authentic Whole Word Bank):** If the exact word exists in `data/words_v2/`, scale dynamically to target line height and anchor to baseline.
2. **Case 1.5 (Hybrid AI Synthesis):** If not in word bank, call `HybridAIProvider.synthesize_word()`:
   - Tries Gemini API first if configured.
   - Automatically falls back to Ollama `gemma4:12b` if Gemini is unconfigured or fails.
   - Validates generated image with ink-quality gate (coverage $\ge 3\%$, non-zero alpha variance).
3. **Case 2 (Authentic Glyph Assembly):** If synthesis is unavailable or fails quality check, assemble character-by-character using verified, non-fragmented glyphs from `data/glyphs_v2/` with proportional kerning.

---

### 3.4 Configuration Management & API Endpoints

#### Configuration File: `data/ai_config.json`
```json
{
  "mode": "hybrid",
  "ollama": {
    "endpoint": "http://localhost:11434",
    "model": "gemma4:12b",
    "timeout_sec": 60
  },
  "gemini": {
    "api_key": "",
    "model": "gemini-2.0-flash",
    "timeout_sec": 15
  }
}
```

#### New & Updated API Endpoints in `app.py`:
- `GET /api/ai/config` -> Returns current configuration and provider availability.
- `POST /api/ai/config` -> Updates Gemini API key, selected Ollama model, or operating mode (`hybrid` vs `ollama_only`).
- `GET /api/ai/health` -> Runs live health checks on both Ollama and Gemini API.
- `GET /api/model-status` -> Updated to reflect:
  - Tier 1: Hybrid AI Synthesis (Ollama `gemma4:12b` + Gemini API)
  - Tier 2: Ollama Vision Extraction (`gemma4:12b`)
  - Tier 3: Procedural Authentic Glyph Engine

---

### 3.5 Frontend UI Enhancements (`static/index.html`)
1. **AI Engine Control Panel:**
   - Displays active engine status pill in the top header:
     - `🟢 Hybrid: Ollama (gemma4:12b) + Gemini API`
     - `🔵 Local: Ollama (gemma4:12b)`
   - Modal/Flyout for AI Settings:
     - Ollama model dropdown (defaults to `gemma4:12b`).
     - Gemini API Key input field with secure hide/show toggle and test connection button.
     - Live ping status indicators for local server and cloud API.

---

## 4. Verification & Testing Strategy

1. **Ollama Integration Test:**
   - Verify `OllamaProvider` connects to `http://localhost:11434`.
   - Confirm `gemma4:12b` generates valid responses within timeout.
2. **Gemini API Integration Test:**
   - Verify `GeminiProvider` handles invalid keys gracefully without crashing.
   - Verify automatic fallback to `OllamaProvider` when Gemini key is empty or network fails.
3. **Extraction & Quality Gate Test:**
   - Test vision extraction on sample note page `data/scans/notes_page_1.jpg`.
   - Verify that all extracted word crops in `data/words_v2/` have width > 20px and zero 6-pixel slivers.
4. **End-to-End Render Test:**
   - Generate document with prompt *"Ow do you do"*.
   - Verify generated preview renders cleanly on A4 canvas without overlapping scribbles or missing glyph errors.

---

## 5. Success Criteria
- [x] Zero 6-pixel broken glyph crops in extraction bank.
- [x] Local Ollama `gemma4:12b` active on GPU.
- [x] Seamless fallback to Ollama when Gemini API is unconfigured.
- [x] Document preview renders clean, naturally proportioned, legible handwriting.
