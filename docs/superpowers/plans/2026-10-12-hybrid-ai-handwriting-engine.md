# Hybrid AI Handwriting Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate a production-grade Hybrid AI subsystem using local Ollama (`gemma4:12b` default on NVIDIA RTX 3070 GPU) and Google Gemini API (cloud) to eliminate fragmented 6px crop scribbles and deliver realistic, beautifully aligned handwritten assignments.

**Architecture:** A unified `HybridAIProvider` coordinates two backends: local `OllamaProvider` (`gemma4:12b`) performs vision-guided note extraction and offline synthesis; `GeminiProvider` performs fast render-time word synthesis when configured, with automatic fallback to Ollama when unconfigured or offline. A clean `VisionExtractor` stores whole cursive words and verified single glyphs, completely banning broken 6px vertical slices.

**Tech Stack:** Python 3.12+, FastAPI, PyTorch (GPU), Pillow (PIL), Requests / HTTPX, Ollama REST API, Google Gemini REST API, Alpine.js / Tailwind (static SPA).

**Spec:** `docs/superpowers/specs/2026-10-12-hybrid-ai-handwriting-engine-design.md`

## Global Constraints
- Ollama endpoint defaults strictly to `http://localhost:11434`.
- Default Ollama model is explicitly `gemma4:12b`.
- All crop validation strictly rejects any crop under 14px in width or height (unless punctuation like `.` or `,`).
- Zero synthetic system font fallbacks in document layout: only authentic crops and neural synthesis allowed.
- Gemini API key must be persistent in `data/ai_config.json` and configurable via REST API.

## Review Focus
1. **Ollama connection timeout / cold start:** The provider must set a 60-second read timeout for Ollama `gemma4:12b` loading into VRAM, and catch connection errors cleanly.
2. **Missing or invalid Gemini API key:** The system must never crash if the Gemini API key is blank, invalid, or quota-exhausted; it must seamlessly fall back to local Ollama.
3. **Empty or blank model response:** If the AI model returns empty text or blank pixels, the ink quality gate must reject the output and fall back to authentic glyph assembly (Case 2).
4. **Scanned page with noise / stray marks:** The vision extractor must filter out bounding boxes with absurd aspect ratios ($\text{width} / \text{height} < 0.2$ for multi-character words).
5. **Special characters / punctuation:** Standalone punctuation marks (`.`, `,`, `-`, `→`) must be preserved and not discarded by the minimum size gate.

---

### Task 1: Core AI Provider Module (`engine/ai_provider.py`)

**Files:**
- Create: `engine/ai_provider.py`
- Test: `tests/test_ai_provider.py`

**Interfaces:**
- Produces:
  - `class AIProvider(ABC)`
  - `class OllamaProvider(AIProvider)`
  - `class GeminiProvider(AIProvider)`
  - `class HybridAIProvider(AIProvider)`
  - Methods: `check_health() -> Dict[str, Any]`, `transcribe_and_locate(img: Image.Image) -> List[Dict[str, Any]]`, `synthesize_word(word: str, target_height: int) -> Optional[Image.Image]`

- [ ] **Step 1: Write the failing unit tests for AI providers**
  Create `tests/test_ai_provider.py` asserting `OllamaProvider` configures `gemma4:12b`, `GeminiProvider` handles empty keys without crashing, and `HybridAIProvider` falls back to Ollama when Gemini is unavailable.
- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_ai_provider.py -v` (Expected: FAIL, module not found).
- [ ] **Step 3: Implement `engine/ai_provider.py`**
  Implement base `AIProvider`, `OllamaProvider` connecting to `http://localhost:11434` with model `gemma4:12b`, `GeminiProvider` querying Google GenAI REST endpoint, and `HybridAIProvider` orchestrating the dual pipeline with automatic fallback.
- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_ai_provider.py -v` (Expected: PASS).
- [ ] **Step 5: Commit**
  ```bash
  git add engine/ai_provider.py tests/test_ai_provider.py
  git commit -m "feat(ai): implement HybridAIProvider with Ollama gemma4:12b and Gemini API"
  ```

---

### Task 2: Vision-Guided Note Extractor (`engine/vision_extractor.py`)

**Files:**
- Create: `engine/vision_extractor.py`
- Modify: `engine/extractor.py:20-60`
- Test: `tests/test_vision_extractor.py`

**Interfaces:**
- Consumes: `HybridAIProvider.transcribe_and_locate(img: Image.Image)`
- Produces:
  - `class VisionNoteExtractor`
  - `extract_from_page(img_bgr: np.ndarray, binary: np.ndarray) -> Dict[str, Any]`
  - Output directories: `data/words_v2/` (whole authentic words), `data/glyphs_v2/` (clean isolated glyphs)

- [ ] **Step 1: Write unit tests for crop size gating and word saving**
  Create `tests/test_vision_extractor.py` ensuring any crop < 14px is rejected, whole words are saved with alpha ink masks, and valid bounding boxes populate `data/words_v2/`.
- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_vision_extractor.py -v` (Expected: FAIL).
- [ ] **Step 3: Implement `engine/vision_extractor.py`**
  Implement `VisionNoteExtractor` using line segmentation and provider-driven word bounding boxes, with strict aspect-ratio and size validation.
- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_vision_extractor.py -v` (Expected: PASS).
- [ ] **Step 5: Commit**
  ```bash
  git add engine/vision_extractor.py tests/test_vision_extractor.py
  git commit -m "feat(extractor): implement vision-guided word and glyph extraction"
  ```

---

### Task 3: Layout Engine Integration with Hybrid AI Synthesis (`engine/layout.py`)

**Files:**
- Modify: `engine/layout.py:380-515`
- Test: `tests/test_layout_hybrid.py`

**Interfaces:**
- Consumes: `HybridAIProvider.synthesize_word(word: str, target_height: int)`
- Produces:
  - Updated `UnruledPageLayout._try_neural_word(clean_word, extra_scale)` delegating to `HybridAIProvider`.
  - Seamless Case 1 (whole words) -> Case 1.5 (Hybrid AI) -> Case 2 (authentic glyph assembly).

- [ ] **Step 1: Write unit test for layout word synthesis and fallback**
  Create `tests/test_layout_hybrid.py` testing that `_generate_word_image` queries the hybrid provider when a word is missing from the bank, and uses authentic glyph assembly if synthesis fails.
- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_layout_hybrid.py -v` (Expected: FAIL).
- [ ] **Step 3: Update `engine/layout.py`**
  Wire `HybridAIProvider` into `UnruledPageLayout`, replacing the old toy PyTorch generator hook with the hybrid synthesizer.
- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_layout_hybrid.py -v` (Expected: PASS).
- [ ] **Step 5: Commit**
  ```bash
  git add engine/layout.py tests/test_layout_hybrid.py
  git commit -m "feat(layout): integrate HybridAIProvider into word generation pipeline"
  ```

---

### Task 4: Configuration API & Health Check Endpoints (`app.py`)

**Files:**
- Modify: `app.py:100-145, 290-340`
- Create: `data/ai_config.json`
- Test: `tests/test_api_ai.py`

**Interfaces:**
- Produces:
  - `GET /api/ai/config` -> returns active provider, model (`gemma4:12b`), and Gemini status
  - `POST /api/ai/config` -> updates configuration JSON
  - `GET /api/ai/health` -> verifies Ollama and Gemini connectivity
  - Updated `/api/model-status` reporting Tier 1 as Hybrid AI Engine

- [ ] **Step 1: Write API tests for AI config and health endpoints**
  Create `tests/test_api_ai.py` testing GET/POST config and health ping responses.
- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_api_ai.py -v` (Expected: FAIL).
- [ ] **Step 3: Implement endpoints in `app.py`**
  Add endpoints and persist configuration to `data/ai_config.json`.
- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_api_ai.py -v` (Expected: PASS).
- [ ] **Step 5: Commit**
  ```bash
  git add app.py tests/test_api_ai.py data/ai_config.json
  git commit -m "feat(api): add AI provider configuration and health check endpoints"
  ```

---

### Task 5: Web UI AI Controls & Status Indicators (`static/index.html`)

**Files:**
- Modify: `static/index.html`

**Interfaces:**
- Consumes: `/api/ai/config`, `/api/ai/health`, `/api/model-status`
- Produces:
  - AI Engine status badge in the top navigation
  - Settings flyout / modal for Ollama model (`gemma4:12b`) and Gemini API Key configuration
  - Visual status pill indicating `Hybrid (Ollama + Gemini)` vs `Local Only (Ollama)`

- [ ] **Step 1: Add AI Engine settings modal and header badge in `static/index.html`**
  Add Alpine.js reactive state: `aiProvider`, `ollamaModel`, `geminiKey`, `aiStatus`.
- [ ] **Step 2: Connect frontend methods to `/api/ai/config` and `/api/ai/health`**
  Implement `loadAIConfig()`, `saveAIConfig()`, and `testAIHealth()`.
- [ ] **Step 3: Verify frontend markup and reactivity**
  Run: `python -c "assert 'ai-config' in open('static/index.html', encoding='utf-8').read() or 'gemma4' in open('static/index.html', encoding='utf-8').read()"`
- [ ] **Step 4: Commit**
  ```bash
  git add static/index.html
  git commit -m "feat(ui): add AI provider settings modal and status badges"
  ```

---

### Task 6: End-to-End Assignment Generation & Document Verification

**Files:**
- Test: `scratch/verify_e2e_render.py`

- [ ] **Step 1: Create verification script `scratch/verify_e2e_render.py`**
  Generates a full document containing *"Ow do you do"*, verifies no 6px fragments are placed, confirms word scaling is proportional, and saves a preview PNG.
- [ ] **Step 2: Execute end-to-end verification**
  Run: `python scratch/verify_e2e_render.py`
  Expected: Clean A4 render generated with legible, beautifully proportioned handwriting.
- [ ] **Step 3: Commit final verification**
  ```bash
  git add scratch/verify_e2e_render.py
  git commit -m "test: verify end-to-end assignment generation with hybrid AI engine"
  ```
