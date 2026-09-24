"""
Train VATr model on user's extracted word crops using NVIDIA RTX 3070 CUDA GPU.
"""

import json
import os
import torch
from PIL import Image
from engine.vatr_synthesizer import HandwritingSynthesizer

def main():
    print("=" * 60)
    print("  VATr Neural Synthesis Training on NVIDIA GPU")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Initial VRAM Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    index_path = "data/words/words_index.json"
    if not os.path.exists(index_path):
        print("Words index not found! Run corpus_builder first.")
        return

    with open(index_path, "r", encoding="utf-8") as f:
        words_data = json.load(f)

    print(f"Loaded {len(words_data)} word metadata entries.")

    word_images = []
    word_labels = []

    for item in words_data:
        path = item["path"]
        label = item["label"].strip()
        # Skip arrow bullet in word dataset
        if label == "→" or len(label) < 2:
            continue
        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGBA")
                word_images.append(img)
                word_labels.append(label)
            except Exception:
                pass

    print(f"Prepared {len(word_images)} valid word samples for training.")

    synthesizer = HandwritingSynthesizer()
    print("Starting neural style fine-tuning (30 epochs)...")

    # Set style references first
    synthesizer.set_style_references(word_images[:15])

    # Train on GPU
    result = synthesizer.train_on_word_crops(word_images, word_labels, epochs=30, lr=1e-3)
    print("Training result:", result)

    if torch.cuda.is_available():
        print(f"Peak VRAM used: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    print(f"Checkpoint saved to: {synthesizer.CHECKPOINT_PATH}")
    print(f"VATr Synthesizer Tier is now: {synthesizer.tier} (Tier 1 = Neural Synthesis Active)")

if __name__ == "__main__":
    main()
