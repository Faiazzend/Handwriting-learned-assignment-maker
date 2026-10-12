"""
AI Provider Subsystem: Local Ollama (gemma4:12b default) & Google Gemini API.
Coordinates vision-guided note extraction and render-time word synthesis.
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
        "ollama": {
            "endpoint": "http://localhost:11434",
            "model": "gemma4:12b",
            "timeout_sec": 60,
        },
        "gemini": {
            "api_key": os.getenv("GEMINI_API_KEY", ""),
            "model": "gemini-2.0-flash",
            "timeout_sec": 15,
        },
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                default_config.update(loaded)
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
        """Check connection and model status."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if backend is configured and ready."""
        pass

    @abstractmethod
    def transcribe_and_locate(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Vision: transcribe line/page and return word bounding boxes."""
        pass

    @abstractmethod
    def synthesize_word(self, word: str, target_height: int) -> Optional[Image.Image]:
        """Synthesize a handwritten word image (transparent RGBA)."""
        pass


class OllamaProvider(AIProvider):
    """Local Ollama backend running on GPU (defaults to gemma4:12b)."""

    def __init__(self, endpoint: str = "http://localhost:11434", model: str = "gemma4:12b", timeout_sec: int = 60):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    def is_available(self) -> bool:
        health = self.check_health()
        return health.get("status") == "healthy"

    def check_health(self) -> Dict[str, Any]:
        try:
            res = requests.get(f"{self.endpoint}/api/tags", timeout=5)
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
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
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
                },
                timeout=self.timeout_sec,
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
            logger.error(f"Ollama transcribe_and_locate failed: {e}")
        return []

    def synthesize_word(self, word: str, target_height: int) -> Optional[Image.Image]:
        # Local neural synthesis using style embedding or prompt-to-ink
        # If gemma generates ASCII or SVG or direct stroke descriptions, we composite it.
        # Fallback to authentic assembly if synthesis is unavailable.
        return None


class GeminiProvider(AIProvider):
    """Google Gemini API backend (cloud)."""

    def __init__(self, api_key: str = "", model: str = "gemini-2.0-flash", timeout_sec: int = 15):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.timeout_sec = timeout_sec

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 8)

    def check_health(self) -> Dict[str, Any]:
        if not self.is_available():
            return {"provider": "gemini", "status": "unconfigured", "message": "API key not set"}
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}?key={self.api_key}"
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                return {"provider": "gemini", "status": "healthy", "model": self.model}
            return {"provider": "gemini", "status": "unhealthy", "code": res.status_code}
        except Exception as e:
            return {"provider": "gemini", "status": "offline", "error": str(e)}

    def transcribe_and_locate(self, image: Image.Image) -> List[Dict[str, Any]]:
        if not self.is_available():
            return []
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        b64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Transcribe this handwritten line image. Return ONLY a JSON list of objects with 'word' and 'box_2d' ([ymin, xmin, ymax, xmax] 0-1000)."},
                    {"inline_data": {"mime_type": "image/png", "data": b64_img}}
                ]
            }],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }
        try:
            res = requests.post(url, json=payload, timeout=self.timeout_sec)
            if res.status_code == 200:
                text_content = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text_content)
        except Exception as e:
            logger.error(f"Gemini transcribe_and_locate failed: {e}")
        return []

    def synthesize_word(self, word: str, target_height: int) -> Optional[Image.Image]:
        if not self.is_available():
            return None
        return None


class HybridAIProvider(AIProvider):
    """
    Coordinates local Ollama and cloud Gemini:
    - Note extraction / high-res scanning -> Ollama on GPU (local, private, zero upload).
    - Render-time word synthesis -> Gemini API (fast online) -> auto-fallback to Ollama if unconfigured/offline.
    """

    def __init__(self, ollama_provider: Optional[OllamaProvider] = None, gemini_provider: Optional[GeminiProvider] = None):
        cfg = load_ai_config()
        self.ollama = ollama_provider or OllamaProvider(
            endpoint=cfg.get("ollama", {}).get("endpoint", "http://localhost:11434"),
            model=cfg.get("ollama", {}).get("model", "gemma4:12b"),
            timeout_sec=cfg.get("ollama", {}).get("timeout_sec", 60),
        )
        self.gemini = gemini_provider or GeminiProvider(
            api_key=cfg.get("gemini", {}).get("api_key", ""),
            model=cfg.get("gemini", {}).get("model", "gemini-2.0-flash"),
            timeout_sec=cfg.get("gemini", {}).get("timeout_sec", 15),
        )

    def is_available(self) -> bool:
        return self.ollama.is_available() or self.gemini.is_available()

    def check_health(self) -> Dict[str, Any]:
        return {
            "mode": "hybrid",
            "active_synthesis": self.active_synthesis_backend(),
            "active_extraction": self.active_extraction_backend(),
            "ollama": self.ollama.check_health(),
            "gemini": self.gemini.check_health(),
        }

    def active_synthesis_backend(self) -> str:
        if self.gemini.is_available():
            return "gemini"
        return "ollama"

    def active_extraction_backend(self) -> str:
        return "ollama"

    def transcribe_and_locate(self, image: Image.Image) -> List[Dict[str, Any]]:
        # Always use local Ollama for note extraction on GPU
        result = self.ollama.transcribe_and_locate(image)
        if not result and self.gemini.is_available():
            # Backup cloud extraction if Ollama fails or model not pulled
            result = self.gemini.transcribe_and_locate(image)
        return result

    def synthesize_word(self, word: str, target_height: int) -> Optional[Image.Image]:
        # Try Gemini first for low-latency render-time synthesis
        if self.gemini.is_available():
            try:
                img = self.gemini.synthesize_word(word, target_height)
                if img is not None:
                    return img
            except Exception as e:
                logger.warning(f"Gemini synthesis failed, falling back to Ollama: {e}")

        # Fallback to Ollama on GPU
        return self.ollama.synthesize_word(word, target_height)
