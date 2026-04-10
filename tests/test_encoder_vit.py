import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models.encoder_vit import vit_base_pretrained, vit_small_pretrained


def test_vit_base_pretrained():
    """ViT-B/16 pretrained: patch tokens (B, 196, 768) → decoder memory."""
    print("Loading pretrained ViT-B/16 weights...")
    model = vit_base_pretrained(return_patch_tokens=True)
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        tokens = model(x)   # (B, 196, 768)

    print(f"Input  shape : {x.shape}")
    print(f"Token  shape : {tokens.shape}  (B, num_patches, embed_dim)  <- feed to decoder")
    assert tokens.shape == (batch_size, 196, 768), f"Got {tokens.shape}"
    print("test_vit_base_pretrained passed.")


def test_vit_small_pretrained():
    """ViT-S/16 pretrained: patch tokens (B, 196, 384) → decoder memory."""
    print("Loading pretrained ViT-S/16 weights...")
    model = vit_small_pretrained(return_patch_tokens=True)
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        tokens = model(x)   # (B, 196, 384)

    print(f"Input  shape : {x.shape}")
    print(f"Token  shape : {tokens.shape}  (B, num_patches, embed_dim)  <- feed to decoder")
    assert tokens.shape == (batch_size, 196, 384), f"Got {tokens.shape}"
    print("test_vit_small_pretrained passed.")


if __name__ == "__main__":
    test_vit_base_pretrained()
    test_vit_small_pretrained()
