"""
Compositor Engine: Ink Shading, Paper Texture, and CamScanner Realism Effects
Simulates physical ballpoint/gel pen ink pressure, paper grain, lighting gradients,
and CamScanner contrast post-processing.
"""

import random
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

class CamScannerCompositor:
    INK_PALETTES = {
        "Royal Blue Ballpoint": (25, 55, 145),
        "Dark Gel Blue": (18, 35, 95),
        "Classic Black": (28, 28, 32),
        "Fountain Blue-Black": (20, 40, 75)
    }

    def __init__(self, page_width=2480, page_height=3508):
        self.page_width = page_width
        self.page_height = page_height

    def render_page(self, placed_glyphs, ink_type="Royal Blue Ballpoint", camscanner_intensity=0.85):
        """
        Renders a full page of placed glyphs with physical ink physics and CamScanner scanning effects.
        """
        # 1. Base paper canvas (warm off-white typical of physical notebook paper)
        paper = np.ones((self.page_height, self.page_width, 3), dtype=np.uint8)
        paper[:, :] = [248, 247, 244]  # Off-white BGR

        # 2. Add subtle natural paper grain noise
        grain = np.random.normal(0, 3, (self.page_height, self.page_width)).astype(np.int16)
        paper = np.clip(paper.astype(np.int16) + grain[:, :, np.newaxis], 0, 255).astype(np.uint8)

        # 3. Create ink mask layer
        ink_layer = Image.new("RGBA", (self.page_width, self.page_height), (0, 0, 0, 0))
        target_color = self.INK_PALETTES.get(ink_type, (25, 55, 145))

        for item in placed_glyphs:
            glyph_img = item["image"]
            gx, gy = item["x"], item["y"]

            # Bounds clamping — prevent alpha_composite crash
            gw, gh = glyph_img.size
            if gx < 0:
                glyph_img = glyph_img.crop((-gx, 0, gw, gh))
                gx = 0
            if gy < 0:
                glyph_img = glyph_img.crop((0, -gy, glyph_img.width, glyph_img.height))
                gy = 0
            gw, gh = glyph_img.size
            if gx + gw > self.page_width:
                glyph_img = glyph_img.crop((0, 0, self.page_width - gx, gh))
            if gy + gh > self.page_height:
                glyph_img = glyph_img.crop((0, 0, glyph_img.width, self.page_height - gy))
            if glyph_img.width <= 0 or glyph_img.height <= 0:
                continue

            # Tint glyph with selected ink color & per-pixel pressure variation
            glyph_np = np.array(glyph_img)
            if glyph_np.shape[2] == 4:
                alpha = glyph_np[:, :, 3].astype(np.float32)

                # Per-pixel pressure: heavier at left (pen-down), lighter at right (pen-lift)
                gh_px, gw_px = alpha.shape
                start_p = random.uniform(0.90, 1.0)
                end_p = random.uniform(0.72, 0.88)
                h_gradient = np.linspace(start_p, end_p, gw_px)
                # Add micro-noise for natural variation
                noise = np.random.normal(0, 0.02, (gh_px, gw_px)).astype(np.float32)
                pressure_mask = np.clip(np.tile(h_gradient, (gh_px, 1)) + noise, 0.6, 1.0)

                # Tint RGB channels with ink color
                tinted = np.zeros_like(glyph_np)
                tinted[:, :, 0] = target_color[0]
                tinted[:, :, 1] = target_color[1]
                tinted[:, :, 2] = target_color[2]
                tinted[:, :, 3] = (alpha * pressure_mask).astype(np.uint8)
                
                tinted_pil = Image.fromarray(tinted, mode="RGBA")
                ink_layer.alpha_composite(tinted_pil, dest=(gx, gy))

        # 4. Composite ink onto paper
        paper_pil = Image.fromarray(cv2.cvtColor(paper, cv2.COLOR_BGR2RGB)).convert("RGBA")
        combined = Image.alpha_composite(paper_pil, ink_layer).convert("RGB")
        combined_cv = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

        # 5. CamScanner Magic Color & Lighting Simulation
        if camscanner_intensity > 0.05:
            combined_cv = self._apply_camscanner_effects(combined_cv, intensity=camscanner_intensity)

        return Image.fromarray(cv2.cvtColor(combined_cv, cv2.COLOR_BGR2RGB))

    def _apply_camscanner_effects(self, img_bgr, intensity=0.85):
        """
        Emulates CamScanner phone capture:
        - Natural phone camera corner shadow / ambient lighting gradient
        - High-contrast 'Magic Color' filter that cleans page background while sharpening ink
        """
        h, w = img_bgr.shape[:2]

        # A. Lighting gradient (phone shadow simulation)
        # Randomize shadow corner (top-left or bottom-right typical of phone camera angle)
        X, Y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        vignette = 1.0 - (0.12 * intensity * (X * 0.7 + Y * 0.3))
        vignette = np.clip(vignette, 0.7, 1.0)

        shaded = (img_bgr.astype(np.float32) * vignette[:, :, np.newaxis]).astype(np.uint8)

        # B. CamScanner 'Magic Color' filter
        # Split into LAB color space to boost luminance contrast while preserving ink chroma
        lab = cv2.cvtColor(shaded, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=1.8 * intensity, tileGridSize=(16, 16))
        l_clahe = clahe.apply(l)

        # Blend CLAHE result with original L
        l_final = cv2.addWeighted(l_clahe, intensity, l, 1.0 - intensity, 0)
        lab_enhanced = cv2.merge((l_final, a, b))
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # C. Subtle unsharp mask to emulate CamScanner edge sharpening
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.3, gaussian, -0.3, 0)

        return sharpened
