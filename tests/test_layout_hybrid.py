import pytest
from PIL import Image
from unittest.mock import MagicMock
from engine.ai_provider import HybridAIProvider
from engine.layout import UnruledPageLayout


def test_layout_delegates_to_hybrid_ai_synthesis():
    mock_hybrid = MagicMock(spec=HybridAIProvider)
    # Return a realistic synthetic word image with visible ink
    test_img = Image.new("RGBA", (220, 55), (0, 0, 0, 0))
    # Fill middle region with solid ink
    for x in range(20, 200):
        for y in range(15, 45):
            test_img.putpixel((x, y), (30, 30, 35, 255))
    mock_hybrid.synthesize_word.return_value = test_img

    # Empty glyph bank and word bank so it must use AI synthesis
    layout = UnruledPageLayout(
        glyph_bank={},
        synthesizer=mock_hybrid,
        words_index_path="data/words_v2/words_index.json"
    )

    res = layout._try_neural_word("NovelWord", extra_scale=1.0)
    assert res is not None
    assert mock_hybrid.synthesize_word.called
    assert res.size[1] == 55
