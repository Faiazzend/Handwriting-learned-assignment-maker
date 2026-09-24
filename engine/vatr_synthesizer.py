"""
VATr Handwriting Synthesizer: Neural word-level handwriting generation.
Uses the Visual Appearance Transformer (VATr) architecture for few-shot
style-conditioned handwriting synthesis.

Falls back to the procedural glyph engine when VATr is not available.

Architecture:
    Content Encoder (text → feature vectors)
    + Style Encoder (reference word images → style embedding)
    → Transformer Generator → realistic handwritten word images
"""

import os
import logging
import random
import numpy as np

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed. VATr synthesis unavailable.")

from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────────────────
# VATr Model Architecture (Inference-only minimal implementation)
# ──────────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=256):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class StyleEncoder(nn.Module):
        """Extracts a fixed-length style embedding from reference handwriting images."""
        def __init__(self, style_dim=256):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc = nn.Linear(256, style_dim)

        def forward(self, x):
            # x: (B, N_ref, 1, H, W) → aggregate over N_ref reference images
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            feats = self.conv(x).squeeze(-1).squeeze(-1)  # (B*N, 256)
            feats = self.fc(feats)  # (B*N, style_dim)
            feats = feats.view(B, N, -1).mean(dim=1)  # Average over references
            return feats

    class ContentEncoder(nn.Module):
        """Encodes text content into a sequence of content feature vectors."""
        CHARSET = list(" !\"#$%&'()*+,-./0123456789:;<=>?@"
                       "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
                       "abcdefghijklmnopqrstuvwxyz{|}~")

        def __init__(self, d_model=256, max_word_len=32):
            super().__init__()
            self.char_to_idx = {c: i + 1 for i, c in enumerate(self.CHARSET)}
            self.embedding = nn.Embedding(len(self.CHARSET) + 1, d_model, padding_idx=0)
            self.pos_enc = PositionalEncoding(d_model, max_len=max_word_len)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        def encode_text(self, text, device):
            indices = [self.char_to_idx.get(c, 0) for c in text[:32]]
            return torch.tensor([indices], dtype=torch.long, device=device)

        def forward(self, token_ids):
            emb = self.embedding(token_ids)
            emb = self.pos_enc(emb)
            return self.transformer(emb)

    class Generator(nn.Module):
        """Transformer-based generator that produces word images from content + style."""
        def __init__(self, d_model=256, img_height=64, max_img_width=256):
            super().__init__()
            self.img_height = img_height
            self.max_img_width = max_img_width
            self.style_proj = nn.Linear(256, d_model)

            decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

            # Upsample from transformer output to image
            self.to_image = nn.Sequential(
                nn.Linear(d_model, 512),
                nn.ReLU(),
                nn.Linear(512, img_height * 4),  # 4 width pixels per token
                nn.Sigmoid()
            )

        def forward(self, content_features, style_embedding):
            B = content_features.size(0)
            style_proj = self.style_proj(style_embedding).unsqueeze(1)  # (B, 1, d_model)
            style_expanded = style_proj.expand(-1, content_features.size(1), -1)

            # Condition content on style by adding
            conditioned = content_features + style_expanded

            # Self-attention decode
            decoded = self.transformer_decoder(conditioned, content_features)

            # Map to image pixels
            pixels = self.to_image(decoded)  # (B, seq_len, H*4)
            seq_len = decoded.size(1)
            img_width = seq_len * 4
            pixels = pixels.view(B, 1, self.img_height, img_width)
            return pixels

    class VATrModel(nn.Module):
        """Complete VATr model for inference."""
        def __init__(self, style_dim=256, d_model=256, img_height=64):
            super().__init__()
            self.style_encoder = StyleEncoder(style_dim)
            self.content_encoder = ContentEncoder(d_model)
            self.generator = Generator(d_model, img_height)

        def forward(self, text_tokens, style_images):
            style_emb = self.style_encoder(style_images)
            content_feats = self.content_encoder(text_tokens)
            word_img = self.generator(content_feats, style_emb)
            return word_img


# ──────────────────────────────────────────────────────────────────────
# Public Interface: Handwriting Synthesizer
# ──────────────────────────────────────────────────────────────────────

class HandwritingSynthesizer:
    """
    High-level interface for generating handwritten word images.

    Tier 1 (VATr): Neural synthesis with few-shot style transfer
    Tier 3 (Fallback): Procedural glyph compositing from personal bank
    """

    CHECKPOINT_DIR = os.path.join("data", "models")
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "vatr_handwriting.pt")

    def __init__(self, glyph_bank=None):
        self.tier = 3  # Default: procedural
        self.model = None
        self.device = None
        self.style_embedding = None
        self.style_images_tensor = None
        self.glyph_bank = glyph_bank or {}

        self._fallback_font = None
        self._init_fallback_font()

        if _TORCH_AVAILABLE:
            self._try_load_model()

    def _init_fallback_font(self):
        for path in ["C:/Windows/Fonts/segoepr.ttf", "C:/Windows/Fonts/comic.ttf",
                      "C:/Windows/Fonts/arial.ttf"]:
            if os.path.exists(path):
                try:
                    self._fallback_font = ImageFont.truetype(path, size=48)
                    return
                except Exception:
                    pass
        self._fallback_font = ImageFont.load_default()

    def _try_load_model(self):
        """Try to load pretrained VATr model."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = VATrModel()

            if os.path.exists(self.CHECKPOINT_PATH):
                state = torch.load(self.CHECKPOINT_PATH, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                self.model.to(self.device)
                self.model.eval()
                self.tier = 1
                logger.info(f"VATr model loaded on {self.device} (Tier 1 synthesis)")
            else:
                # Model architecture ready but no pretrained weights
                # User needs to train/download checkpoint
                self.model.to(self.device)
                self.model.eval()
                self.tier = 3  # Still procedural until trained
                logger.info("VATr architecture ready. No checkpoint found - using procedural fallback.")

        except Exception as e:
            logger.warning(f"VATr initialization failed: {e}. Using procedural fallback.")
            self.model = None
            self.tier = 3

    def get_status(self):
        return {
            "tier": self.tier,
            "torch_available": _TORCH_AVAILABLE,
            "model_loaded": self.model is not None,
            "checkpoint_exists": os.path.exists(self.CHECKPOINT_PATH),
            "device": str(self.device) if self.device else "cpu",
            "style_loaded": self.style_embedding is not None,
            "status": "active" if self.tier == 1 else ("available" if _TORCH_AVAILABLE else "unavailable")
        }

    def set_style_references(self, word_images):
        """
        Feed 10-15 cropped word images from the user's handwriting.
        These are used as style references for neural synthesis.

        Args:
            word_images: list of PIL Images (cropped words from user's notes)
        """
        if not _TORCH_AVAILABLE or self.model is None:
            logger.info("Style references stored for future use (VATr not active)")
            return

        try:
            processed = []
            for img in word_images[:15]:
                gray = img.convert("L")
                gray = gray.resize((128, 64), Image.Resampling.LANCZOS)
                arr = np.array(gray, dtype=np.float32) / 255.0
                processed.append(arr)

            # Stack into (1, N_ref, 1, 64, 128) tensor
            refs = np.stack(processed)
            self.style_images_tensor = torch.tensor(refs, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(self.device)

            with torch.no_grad():
                self.style_embedding = self.model.style_encoder(self.style_images_tensor)

            logger.info(f"Style embedding computed from {len(word_images)} reference words")

        except Exception as e:
            logger.error(f"Style reference processing failed: {e}")

    def generate_word(self, text, target_height=65):
        """
        Generate a handwritten word image in the user's style.

        Args:
            text: The word to render
            target_height: Desired height in pixels

        Returns:
            PIL Image (RGBA) of the handwritten word
        """
        if self.tier == 1 and self.model is not None and self.style_embedding is not None:
            return self._generate_neural(text, target_height)
        else:
            return self._generate_procedural(text, target_height)

    def _generate_neural(self, text, target_height):
        """Generate using VATr neural synthesis."""
        try:
            with torch.no_grad():
                token_ids = self.model.content_encoder.encode_text(text, self.device)
                content_feats = self.model.content_encoder(token_ids)
                word_img_tensor = self.model.generator(content_feats, self.style_embedding)

                # Convert tensor to PIL Image
                img_np = word_img_tensor.squeeze().cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)

                if img_np.ndim == 1:
                    # Reshape based on expected dimensions
                    h = 64
                    w = len(img_np) // h if h > 0 else 1
                    img_np = img_np[:h * w].reshape(h, w)
                elif img_np.ndim == 3:
                    img_np = img_np[0]  # Take first channel

                pil_img = Image.fromarray(img_np, mode="L")

                # Resize to target height
                aspect = pil_img.width / max(pil_img.height, 1)
                new_w = max(int(target_height * aspect), 10)
                pil_img = pil_img.resize((new_w, target_height), Image.Resampling.LANCZOS)

                # Convert to RGBA with ink as alpha
                rgba = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
                pixels = np.array(pil_img)
                rgba_np = np.array(rgba)
                rgba_np[:, :, 0] = 30
                rgba_np[:, :, 1] = 30
                rgba_np[:, :, 2] = 35
                rgba_np[:, :, 3] = 255 - pixels  # Invert: dark pixels = visible ink
                return Image.fromarray(rgba_np, mode="RGBA")

        except Exception as e:
            logger.error(f"Neural generation failed for '{text}': {e}")
            return self._generate_procedural(text, target_height)

    def _generate_procedural(self, text, target_height):
        """Fallback: Generate using glyph bank compositing or synthetic font."""
        total_width = 0
        char_images = []

        for i, ch in enumerate(text):
            variants = self.glyph_bank.get(ch, [])
            glyph_img = None

            if variants:
                selected = random.choice(variants)
                path = selected.get("path")
                if path and os.path.exists(path):
                    try:
                        glyph_img = Image.open(path).convert("RGBA")
                    except Exception:
                        pass

            if glyph_img is None:
                # Synthetic fallback
                glyph_img = self._make_synthetic_char(ch)

            # Scale to target height
            if glyph_img.height != target_height and glyph_img.height > 0:
                aspect = glyph_img.width / glyph_img.height
                new_w = max(int(target_height * aspect), 8)
                glyph_img = glyph_img.resize((new_w, target_height), Image.Resampling.LANCZOS)

            char_images.append(glyph_img)
            total_width += glyph_img.width + random.randint(1, 3)

        # Composite all characters side by side
        word_img = Image.new("RGBA", (total_width + 5, target_height), (0, 0, 0, 0))
        x_cursor = 0
        for ci in char_images:
            word_img.alpha_composite(ci, dest=(x_cursor, 0))
            x_cursor += ci.width + random.randint(1, 3)

        return word_img

    def _make_synthetic_char(self, ch):
        """Create a synthetic character glyph using system fonts."""
        temp = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        d = ImageDraw.Draw(temp)
        bbox = d.textbbox((0, 0), ch, font=self._fallback_font)
        w = max(bbox[2] - bbox[0] + 8, 15)
        h = max(bbox[3] - bbox[1] + 8, 20)

        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((4 - bbox[0], 4 - bbox[1]), ch, font=self._fallback_font, fill=(30, 30, 35, 255))
        return img

    def train_on_word_crops(self, word_images, word_labels, epochs=50, lr=1e-4):
        """
        Fine-tune the VATr model on user's word crops for personalized synthesis.

        Args:
            word_images: list of PIL Images (cropped handwritten words)
            word_labels: list of str (text content of each word)
            epochs: training iterations
            lr: learning rate
        """
        if not _TORCH_AVAILABLE or self.model is None:
            logger.warning("Cannot train: PyTorch or VATr model not available")
            return {"error": "PyTorch not available"}

        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Pre-process all targets into GPU tensors ONCE
        logger.info(f"Pre-tensorizing {len(word_images)} word images to {self.device}...")
        targets = []
        token_list = []
        for img, label in zip(word_images, word_labels):
            gray = img.convert("L").resize((128, 64), Image.Resampling.BILINEAR)
            t = torch.tensor(
                np.array(gray, dtype=np.float32) / 255.0,
                device=self.device
            ).unsqueeze(0).unsqueeze(0)
            targets.append(t)
            token_list.append(self.model.content_encoder.encode_text(label, self.device))

        # Pre-process fixed style reference tensor (top 10 clean words)
        style_refs = []
        for si in word_images[:10]:
            sg = si.convert("L").resize((128, 64), Image.Resampling.BILINEAR)
            style_refs.append(np.array(sg, dtype=np.float32) / 255.0)
        style_tensor = torch.tensor(
            np.stack(style_refs),
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(2)

        # Compute style embedding once and detach from graph
        with torch.no_grad():
            style_emb = self.model.style_encoder(style_tensor).detach()

        for epoch in range(epochs):
            total_loss = 0.0
            for target, token_ids in zip(targets, token_list):
                content = self.model.content_encoder(token_ids)
                generated = self.model.generator(content, style_emb)
                generated_resized = F.interpolate(generated, size=target.shape[2:], mode="bilinear", align_corners=False)
                loss = F.mse_loss(generated_resized, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg = total_loss / max(len(targets), 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg:.4f}")

        # Save checkpoint
        torch.save(self.model.state_dict(), self.CHECKPOINT_PATH)
        self.model.eval()
        self.tier = 1
        logger.info(f"VATr model trained and saved to {self.CHECKPOINT_PATH}")

        return {"epochs": epochs, "final_loss": total_loss / max(len(word_images), 1)}
