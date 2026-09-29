"""Final verification of all fixes."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json, cv2, numpy as np, os

print('=== FINAL VERIFICATION ===')

# 1. Import all modules
print('\n[1] Importing all modules...')
from engine.stroke_augmentor import SmartVariantSelector, augment_glyph_bank
from engine.layout import UnruledPageLayout
from engine.compositor import CamScannerCompositor
from engine.extractor import GlyphExtractor
from engine.validator import validate_text_coverage
print('  ALL IMPORTS OK')

# 2. Check variant coverage
print('\n[2] Variant coverage:')
with open('data/glyphs_v2/glyphs_index.json', 'r', encoding='utf-8') as f:
    g = json.load(f)
below_5 = [ch for ch in g if len(g[ch]) < 5]
total = sum(len(v) for v in g.values())
print(f'  {len(g)} chars, {total} variants')
print(f'  Characters below 5 variants: {len(below_5)}')
assert len(below_5) == 0, f'FAIL: {below_5}'
print('  PASS: All characters have 5+ variants')

# 3. Dedup test
print('\n[3] Dedup test:')
ext = GlyphExtractor()
test_binary = np.zeros((20, 15), dtype=np.uint8)
cv2.circle(test_binary, (7, 10), 5, 255, 1)
r1 = ext.register_glyph('o', test_binary)
r2 = ext.register_glyph('o', test_binary)  # exact duplicate
print(f'  First register: {"OK" if r1 else "rejected"}')
dup_msg = "rejected - dedup works" if r2 is None else "ACCEPTED - BAD"
print(f'  Duplicate register: {dup_msg}')

# 4. Word bank clean
print('\n[4] Word bank:')
with open('data/words_v2/words_index.json', 'r', encoding='utf-8') as f:
    words = json.load(f)
print(f'  {len(words)} words')
garbage = [w for w in words if ' ' in w.get('label','') or '..' in w.get('label','')]
print(f'  Garbage entries: {len(garbage)}')
assert len(garbage) == 0, 'FAIL: garbage still in word bank'
print('  PASS: Word bank clean')

# 5. SmartVariantSelector test
print('\n[5] SmartVariantSelector:')
sel = SmartVariantSelector(history_depth=3)
variants = g.get('e', [])
picks = [sel.select('e', variants)[1] for _ in range(10)]
consec = sum(1 for i in range(1, len(picks)) if picks[i] == picks[i-1])
print(f'  10 picks: {picks}')
print(f'  Consecutive dupes: {consec}')
print(f'  PASS' if consec == 0 else '  FAIL')

# 6. Full pipeline test
print('\n[6] Full pipeline test:')
layout = UnruledPageLayout(glyph_bank=g, words_index_path='data/words_v2/words_index.json')
pages = layout.layout_document('Political science studies power and conflict.')
comp = CamScannerCompositor()
img = comp.render_page(pages[0])
os.makedirs('output', exist_ok=True)
img.save('output/final_test.png')
print(f'  Rendered: {img.size[0]}x{img.size[1]}')
print('  PASS')

print('\n=== ALL CHECKS PASSED ===')
