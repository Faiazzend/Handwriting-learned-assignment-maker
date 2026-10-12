import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock
from engine.ai_provider import AIProvider
from engine.vision_extractor import VisionNoteExtractor, is_valid_crop_size


def test_is_valid_crop_size_banning_slivers():
    # 6px width slivers must be rejected
    assert is_valid_crop_size(6, 7, "w") is False
    assert is_valid_crop_size(9, 26, "O") is False
    # Valid sizes
    assert is_valid_crop_size(25, 35, "w") is True
    assert is_valid_crop_size(30, 45, "O") is True
    # Punctuation allowed smaller
    assert is_valid_crop_size(6, 8, ".") is True
    assert is_valid_crop_size(6, 12, ",") is True


def test_vision_extractor_rejects_fragment_bounding_boxes():
    mock_provider = MagicMock(spec=AIProvider)
    # Return one invalid 6px fragment and one valid word box
    mock_provider.transcribe_and_locate.return_value = [
        {"word": "sliver", "box_2d": [100, 100, 110, 106]},  # 6px width -> must be rejected
        {"word": "Early", "box_2d": [100, 100, 140, 260]},   # 160px width, 40px height -> valid
    ]

    extractor = VisionNoteExtractor(ai_provider=mock_provider)
    # Create blank canvas image
    img_bgr = np.zeros((500, 500, 3), dtype=np.uint8)
    binary = np.zeros((500, 500), dtype=np.uint8)
    # Add ink into the Early region
    binary[100:140, 100:260] = 255

    results = extractor.process_line_image(img_bgr, binary, (100, 140))
    # 'sliver' should be rejected, 'Early' accepted
    labels = [r["word"] for r in results]
    assert "sliver" not in labels
    assert "Early" in labels
