"""
AI Stroke Augmentor: Generates natural micro-variants from scanned glyphs
using elastic deformation, thickness jitter, and pressure modulation.

Solves the "identical letter" problem where low-variant characters (1-2 variants)
produce visibly mechanical repetition that human eyes catch instantly.
"""

import cv2
import numpy as np
import random
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter


def elastic_deform(binary_crop, alpha=3.0, sigma=1.2):
    """
    Apply elastic deformation to simulate natural handwriting tremor.
    alpha: intensity of displacement
    sigma: smoothness of displacement field
    """
    h, w = binary_crop.shape
    if h < 4 or w < 4:
        return binary_crop

    # Random displacement fields
    dx = gaussian_filter(np.random.randn(h, w) * alpha, sigma, mode='reflect')
    dy = gaussian_filter(np.random.randn(h, w) * alpha, sigma, mode='reflect')

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    indices = [np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)]

    warped = map_coordinates(binary_crop.astype(np.float64), indices, order=1, mode='reflect')
    return np.clip(warped, 0, 255).astype(np.uint8)


def thickness_jitter(binary_crop, mode='random'):
    """
    Randomly thicken or thin strokes via morphological ops.
    Simulates pen pressure variation.
    """
    if mode == 'random':
        mode = random.choice(['thin', 'normal', 'thick'])

    if mode == 'thin':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        return cv2.erode(binary_crop, kernel, iterations=1)
    elif mode == 'thick':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        return cv2.dilate(binary_crop, kernel, iterations=1)
    return binary_crop


def micro_affine(binary_crop, max_rotation=1.5, max_shear=0.04):
    """
    Apply subtle random rotation and shear.
    Simulates wrist angle variation between writing same letter.
    """
    h, w = binary_crop.shape
    if h < 4 or w < 4:
        return binary_crop

    angle = random.uniform(-max_rotation, max_rotation)
    shear_x = random.uniform(-max_shear, max_shear)

    cx, cy = w / 2, h / 2
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))

    M = np.array([
        [cos_a + shear_x * sin_a, -sin_a + shear_x * cos_a, cx - cx * (cos_a + shear_x * sin_a) + cy * (sin_a - shear_x * cos_a)],
        [sin_a, cos_a, cy - cx * sin_a - cy * cos_a]
    ], dtype=np.float32)

    warped = cv2.warpAffine(binary_crop, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


def pressure_gradient(binary_crop, direction='horizontal'):
    """
    Apply writing-direction pressure gradient.
    Start of stroke is heavier (darker), end is lighter.
    Simulates real ballpoint pen physics.
    """
    h, w = binary_crop.shape
    if h < 4 or w < 4:
        return binary_crop

    if direction == 'horizontal':
        # Heavier on left (pen down), lighter on right (pen lift)
        start_pressure = random.uniform(0.95, 1.0)
        end_pressure = random.uniform(0.65, 0.85)
        gradient = np.linspace(start_pressure, end_pressure, w)
        mask = np.tile(gradient, (h, 1))
    else:
        start_pressure = random.uniform(0.9, 1.0)
        end_pressure = random.uniform(0.7, 0.9)
        gradient = np.linspace(start_pressure, end_pressure, h)
        mask = np.tile(gradient, (w, 1)).T

    result = (binary_crop.astype(np.float64) * mask).astype(np.uint8)
    return result


def generate_augmented_variant(binary_crop, intensity='medium'):
    """
    Generate one natural-looking augmented variant from a source glyph.

    intensity: 'light' (subtle), 'medium' (moderate), 'heavy' (aggressive)
    """
    result = binary_crop.copy()

    if intensity == 'light':
        # Only micro-affine + subtle elastic
        result = micro_affine(result, max_rotation=0.8, max_shear=0.02)
        if random.random() > 0.4:
            result = elastic_deform(result, alpha=1.5, sigma=1.0)
    elif intensity == 'medium':
        result = elastic_deform(result, alpha=2.5, sigma=1.2)
        result = micro_affine(result, max_rotation=1.2, max_shear=0.03)
        result = thickness_jitter(result)
        if random.random() > 0.5:
            result = pressure_gradient(result)
    else:  # heavy
        result = elastic_deform(result, alpha=4.0, sigma=1.5)
        result = micro_affine(result, max_rotation=2.0, max_shear=0.05)
        result = thickness_jitter(result)
        result = pressure_gradient(result)

    # Ensure binary mask stays clean
    _, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY)
    return result


def augment_glyph_bank(glyph_index, glyphs_dir, min_variants=5, max_augments_per_source=4):
    """
    Scan glyph bank and generate AI-augmented variants for characters
    with fewer than min_variants variants.

    Returns count of new variants created.
    """
    import os

    created = 0

    for char_label, variants in list(glyph_index.items()):
        valid = [v for v in variants if v.get("path") and os.path.exists(v["path"])]
        if len(valid) >= min_variants:
            continue

        # Need more variants for this character
        needed = min_variants - len(valid)
        sources = valid.copy()
        if not sources:
            continue

        char_dir = os.path.join(glyphs_dir, f"char_{ord(char_label) if len(char_label) == 1 else 'bigram_' + char_label}")
        os.makedirs(char_dir, exist_ok=True)

        for _ in range(needed):
            # Pick random source variant
            src = random.choice(sources)
            src_img = cv2.imread(src["path"], cv2.IMREAD_UNCHANGED)
            if src_img is None or src_img.shape[2] < 4:
                continue

            # Extract alpha channel as binary mask
            alpha = src_img[:, :, 3]

            # Generate augmented variant
            intensity = random.choice(['light', 'medium', 'medium'])
            augmented = generate_augmented_variant(alpha, intensity=intensity)

            # Save as new variant
            var_id = len(glyph_index[char_label]) + 1
            filepath = os.path.join(char_dir, f"{var_id}.png").replace('\\', '/')

            h, w = augmented.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, 0] = 30
            rgba[:, :, 1] = 30
            rgba[:, :, 2] = 35
            rgba[:, :, 3] = augmented
            cv2.imwrite(filepath, rgba)

            # Compute perceptual hash for deduplication
            resized_hash = cv2.resize(augmented, (16, 16), interpolation=cv2.INTER_AREA)
            _, hash_bin = cv2.threshold(resized_hash, 127, 1, cv2.THRESH_BINARY)
            aug_hash = hash_bin.flatten().tolist()

            record = {
                "variant_id": var_id,
                "path": filepath,
                "width": int(w),
                "height": int(h),
                "aspect_ratio": round(float(w) / float(h), 2),
                "augmented": True,
                "source_variant": src.get("variant_id"),
                "intensity": intensity,
                "phash": aug_hash
            }
            glyph_index[char_label].append(record)
            created += 1

    return created


class SmartVariantSelector:
    """
    Selects glyph variants using N-gram awareness to prevent visible repetition.
    Tracks last N variants used per character and avoids them.
    """

    def __init__(self, history_depth=3):
        self.history_depth = history_depth
        self._history = {}  # char -> list of recent variant_ids

    def select(self, char, variants):
        """
        Pick variant avoiding recent history.
        Returns (variant_dict, variant_id).
        """
        if not variants:
            return None, None

        valid = [v for v in variants if v.get("path")]
        if not valid:
            return None, None

        history = self._history.get(char, [])

        # Filter out recently used
        if len(valid) > self.history_depth:
            pool = [v for v in valid if v.get("variant_id") not in history]
            if not pool:
                pool = valid
        else:
            # Not enough variants to avoid all history, just avoid last one
            if len(valid) > 1 and history:
                pool = [v for v in valid if v.get("variant_id") != history[-1]]
                if not pool:
                    pool = valid
            else:
                pool = valid

        selected = random.choice(pool)
        vid = selected.get("variant_id")

        # Update history
        if char not in self._history:
            self._history[char] = []
        self._history[char].append(vid)
        if len(self._history[char]) > self.history_depth:
            self._history[char] = self._history[char][-self.history_depth:]

        return selected, vid

    def reset(self):
        self._history = {}
