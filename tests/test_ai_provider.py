import pytest
from PIL import Image
from unittest.mock import patch, MagicMock
from engine.ai_provider import (
    AIProvider,
    OllamaProvider,
    GeminiProvider,
    HybridAIProvider,
)


def test_ollama_provider_defaults():
    provider = OllamaProvider()
    assert provider.endpoint == "http://localhost:11434"
    assert provider.model == "gemma4:12b"


def test_gemini_provider_unconfigured():
    provider = GeminiProvider(api_key="")
    assert provider.is_available() is False
    # Calling synthesize_word when unconfigured should return None, not raise an exception
    assert provider.synthesize_word("test", 50) is None


def test_hybrid_provider_fallback_to_ollama():
    # Gemini is unconfigured, so Hybrid should fall back to Ollama
    hybrid = HybridAIProvider(
        ollama_provider=OllamaProvider(model="gemma4:12b"),
        gemini_provider=GeminiProvider(api_key=""),
    )
    assert hybrid.active_synthesis_backend() == "ollama"
    assert hybrid.active_extraction_backend() == "ollama"


def test_hybrid_provider_with_gemini_available():
    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.is_available.return_value = True
    test_img = Image.new("RGBA", (100, 50), (0, 0, 0, 255))
    mock_gemini.synthesize_word.return_value = test_img

    hybrid = HybridAIProvider(
        ollama_provider=OllamaProvider(model="gemma4:12b"),
        gemini_provider=mock_gemini,
    )
    assert hybrid.active_synthesis_backend() == "gemini"
    result = hybrid.synthesize_word("Early", 50)
    assert result is not None
    assert result.size == (100, 50)
    mock_gemini.synthesize_word.assert_called_once_with("Early", 50)
