import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models.encoder_vmamba import vanilla_vmamba_small


def test_feature_extraction():
    print("Loading pretrained VMamba-Small weights (downloads ~100MB on first run)...")
    model = vanilla_vmamba_small(pretrained=True)
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        features = model(x)

    # Encoder output: (B, H, W, C)
    #   H = W = 224 / patch_size=4 / 2^3 inter-stage downsamples = 7
    #   C     = dims[-1] = 96 * 2^3 = 768
    B, H, W, C = features.shape
    print(f"Input   shape : {x.shape}")
    print(f"Feature shape : {features.shape}  (B, H, W, C)")
    assert features.shape == (batch_size, 7, 7, 768), f"Got {features.shape}"

    # Decoder input: flatten spatial dims into a token sequence (B, num_tokens, C)
    tokens = features.view(B, H * W, C)
    print(f"Token   shape : {tokens.shape}  (B, num_tokens, C)  <- feed this to your decoder")
    assert tokens.shape == (batch_size, 49, 768)

    print("Test passed.")


if __name__ == "__main__":
    test_feature_extraction()
