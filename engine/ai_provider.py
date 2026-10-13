"""
AI Provider Subsystem: Instant Local RapidOCR, Fast Google Gemini API, & Local Ollama.
Coordinates vision-guided note extraction and render-time word synthesis with fail-fast timeouts.
"""

import os
import io
import json
import base64
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from PIL import Image
import requests

logger = logging.getLogger(__name__)

CONFIG_PATH = "data/ai_config.json"


def load_ai_config() -> Dict[str, Any]:
    default_config = {
        "mode": "hybrid",
        "extraction_mode": "rapidocr",
        "ollama": {
            "endpoint": "http://localhost:11434",
            "model": "qwen3.5:4b",
            "timeout_sec": 30,
        },
        "gemini": {
            "api_key": os.getenv("GEMINI_API_KEY", ""),
            "model": "gemini-3.1-flash-lite",
            "timeout_sec": 4,
        },
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                default_config.update(loaded)
                # Auto-upgrade deprecated Gemini models
                cur_model = default_config.get("gemini", {}).get("model")
                if cur_model in ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-3.5-flash-lite"]:
                    default_config["gemini"]["model"] = "gemini-3.1-flash-lite"
                    save_ai_config(default_config)
                # Ensure timeout is fail-fast (max 4s)
                if default_config.get("gemini", {}).get("timeout_sec", 4) > 5:
                    default_config["gemini"]["timeout_sec"] = 4
                    save_ai_config(default_config)
        except Exception as e:
            logger.warning(f"Failed to read {CONFIG_PATH}: {e}")
    return default_config


def save_ai_config(config: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


class AIProvider(ABC):
    """Abstract Base Class for AI backends."""

    @abstractmethod
    def check_health(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def transcribe_and_locate(self, image: Image.Image) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def synthesize_word(self, word: str, target_height: int) -> Optional[Image.Image]:
        pass


class OllamaProvider(AIProvider):
    """Local Ollama instance running on GPU/CPU."""

    def __init__(self, endpoint: str = "http://localhost:11434", model: str = "qwen3.5:4b", timeout_sec: int = 30):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    def is_available(self) -> bool:
        health = self.check_health()
        return health.get("status") == "healthy"

    def check_health(self) -> Dict[str, Any]:
        try:
            res = requests.get(f"{self.endpoint}/api/tags", timeout=4)
            if res.status_code == 200:
                data = res.json()
                models = [m.get("name") for m in data.get("models", [])]
                model_found = any(self.model in m for m in models)
                return {
                    "provider": "ollama",
                    "status": "healthy" if model_found else "model_missing",
                    "endpoint": self.endpoint,
                    "target_model": self.model,
                    "available_models": models,
                }
            return {"provider": "ollama", "status": "unhealthy", "code": res.status_code}
        except Exception as e:
            return {"provider": "ollama", "status": "offline", "error": str(e)}

    def transcribe_and_locate(self, image: Image.Image) -> List[Dict[str, Any]]:
        # Downscale line image for fast vision inference
        img_for_ai = image.copy()
        if img_for_ai.width > 640 or img_for_ai.height > 64:
            aspect = img_for_ai.width / max(img_for_ai.height, 1)
            target_h = min(img_for_ai.height, 64)
            target_w = min(int(target_h * aspect), 640)
            img_for_ai = img_for_ai.resize((max(target_w, 10), max(target_h, 10)), Image.Resampling.BILINEAR)

        buffered = io.BytesIO()
        img_for_ai.save(buffered, format="JPEG", quality=85)
        b64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = (
            "Analyze this handwritten line image. Return ONLY a valid JSON list of words with their 2D bounding boxes "
            "in normalized 0-1000 coordinates. Schema: [{\"word\": \"text\", \"box_2d\": [ymin, xmin, ymax, xmax]}]. "
            "No commentary or markdown fences."
        )

        try:
            res = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [b64_img],
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 256
                    }
                },
                timeout=min(self.timeout_sec, 8),
            )
            if res.status_code == 200:
                raw_text = res.json().get("response", "").strip()
                try:
                    data = json.loads(raw_text)
                    if isinstance(data, list):
                        return data
                    if isinstance(data, dict) and "words" in data:
                        return data["words"]
                except json.JSONDecodeError:
                    logger.warning(f"Ollama JSON decode failed: {raw_text[:100]}")
        except Exception as e:
            logger.warning(f"Ollama transcribe_and_locate failed: {e}")
        return []

    def synthesize_word(self, word: str, target_height: int) -> Optional[Image.Image]:
        return None


class GeminiProvider(AIProvider):
    """Google Gemini API backend (cloud) with strict 4s fail-fast timeout."""

    def __init__(self, api_key: str = "", model: str = "gemini-3.1-flash-lite", timeout_sec: int = 4):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if model in ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-3.5-flash-lite"]:
            model = "gemini-3.1-flash-lite"
        self.model = model
        self.timeout_sec = min(timeout_sec, 4)  # Hard cap at 4s to prevent UI freeze

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 8)

    def check_health(self) -> Dict[str, Any]:
        if not self.is_available():
            return {"provider": "gemini", "status": "unconfigured", "message": "API key not set"}
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}?key={self.api_key}"
            res = requests.get(url, timeout=3)
            if res.status_code == 200:
                return {"provider": "gemini", "status": "healthy", "model": self.model}
            return {"provider": "gemini", "status": "unhealthy", "code": res.status_code}
        except Exception as e:
            return {"provider": "gemini", "status": "offline", "error": str(e)}

    def transcribe_and_locate(self, image: Image.Image) -> List[Dict[str, Any]]:
        if not self.is_available():
            return []

        # Downscale and compress to lightweight JPEG (~25KB payload instead of ~300KB PNG)
        img_for_ai = image.copy()
        if img_for_ai.width > 1024:
            aspect = img_for_ai.width / max(img_for_ai.height, 1)
            target_h = int(1024 / aspect)
            img_for_ai = img_for_ai.resize((1024, max(target_h, 10)), Image.Resampling.BILINEAR)

        buffered = io.BytesIO()
        img_for_ai.save(buffered, format="JPEG", quality=85)
        b64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = (
            'Identify all handwritten words in this line. Return ONLY a valid JSON list of objects: '
            '[{"word": "text", "box_2d": [ymin, xmin, ymax, xmax]}]. Normalized 0 to 1000 coordinates. No markdown.'
        )
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}
                ]
            }],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }

        # Single fast attempt with 4s timeout (no serial 80-second loop)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        try:
            res = requests.post(url, json=payload, timeout=self.timeout_sec)
            if res.status_code == 200:
                text_content = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(text_content)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    for k in ["words", "result", "items", "detections"]:
                        if k in data and isinstance(data[k], list):
                            return data[k]
            elif res.status_code in [503, 429]:
                logger.info(f"Gemini {self.model} busy (HTTP {res.status_code}), failing fast to local OCR")
        except Exception as e:
            logger.info(f"Gemini attempt timed out or failed ({e}), failing fast to local OCR")
        return []

    def synthesize_word(self, word: str, target_height: int) -> Optional[Image.Image]:
        return None


class HybridAIProvider(AIProvider):
    """
    Coordinates local RapidOCR, local Ollama, and cloud Gemini:
    - mode="rapidocr": Instant local extraction via ONNX (~3s whole page).
    - mode="gemini": Cloud AI vision with 4s fail-fast timeout and instant RapidOCR fallback.
    - mode="ollama": Local LLM vision on GPU/CPU.
    """

    def __init__(self, ollama_provider: Optional[OllamaProvider] = None, gemini_provider: Optional[GeminiProvider] = None):
        cfg = load_ai_config()
        self.ollama = ollama_provider or OllamaProvider(
            endpoint=cfg.get("ollama", {}).get("endpoint", "http://localhost:11434"),
            model=cfg.get("ollama", {}).get("model", "qwen3.5:4b"),
            timeout_sec=cfg.get("ollama", {}).get("timeout_sec", 30),
        )
        self.gemini = gemini_provider or GeminiProvider(
            api_key=cfg.get("gemini", {}).get("api_key", ""),
            model=cfg.get("gemini", {}).get("model", "gemini-3.1-flash-lite"),
            timeout_sec=cfg.get("gemini", {}).get("timeout_sec", 4),
        )

    def is_available(self) -> bool:
        return True  # RapidOCR is always available locally

    def check_health(self) -> Dict[str, Any]:
        return {
            "mode": "hybrid",
            "active_synthesis": self.active_synthesis_backend(),
            "active_extraction": self.active_extraction_backend(),
            "rapidocr": {"provider": "rapidocr", "status": "healthy"},
            "ollama": self.ollama.check_health(),
            "gemini": self.gemini.check_health(),
        }

    def active_synthesis_backend(self) -> str:
        if self.gemini.is_available():
            return "gemini"
        return "ollama"

    def active_extraction_backend(self) -> str:
        cfg = load_ai_config()
        mode = cfg.get("extraction_mode", "rapidocr")
        if mode == "gemini" and self.gemini.is_available():
            return "gemini"
        if mode == "ollama" and self.ollama.is_available():
            return "ollama"
        return "rapidocr"

    def get_extraction_display_name(self) -> str:
        backend = self.active_extraction_backend()
        if backend == "gemini":
            return f"Gemini Cloud ({self.gemini.model})"
        if backend == "ollama":
            return f"Ollama Local ({self.ollama.model})"
        return "RapidOCR Local (Instant ~3s)"

    def transcribe_and_locate(self, image: Image.Image) -> List[Dict[str, Any]]:
        backend = self.active_extraction_backend()

        if backend == "gemini":
            try:
                res = self.gemini.transcribe_and_locate(image)
                if res:
                    return res
            except Exception as e:
                logger.info(f"Gemini transcription error: {e}")
            # Instant fallback to empty so vision_extractor runs local RapidOCR
            return []

        if backend == "ollama":
            return self.ollama.transcribe_and_locate(image)

        # rapidocr mode delegates to local OCR in vision_extractor
        return []

    def synthesize_word(self, word: str, target_height: int) -> Optional[Image.Image]:
        if self.gemini.is_available():
            try:
                img = self.gemini.synthesize_word(word, target_height)
                if img is not None:
                    return img
            except Exception:
                pass
        return self.ollama.synthesize_word(word, target_height)
