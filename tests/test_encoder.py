import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models.encoder import VSSM


def test_feature_extraction():
    # Tiny config for fast CPU testing (same architecture as vanilla_vmamba_small)
    model = VSSM(
        depths=[1, 1, 1, 1], dims=32,
        patch_size=4, in_chans=3,
        ssm_d_state=4, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v0",
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v1", patchembed_version="v1",
        use_checkpoint=False, posembed=False, imgsize=224,
    )
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        features = model(x)

    # Encoder output: (B, H, W, C)
    #   H = W = 224 / (patch_size=4) / 2^3 inter-stage downsamples = 7
    #           4 stages, downsample runs on all but last (i_layer < num_layers-1)
    #   C     = dims[-1] = 32 * 2^3 = 256
    B, H, W, C = features.shape
    print(f"Input   shape : {x.shape}")
    print(f"Feature shape : {features.shape}  (B, H, W, C)")
    assert features.shape == (batch_size, 7, 7, 256)

    # Decoder input: flatten spatial dims into a token sequence (B, num_tokens, C)
    tokens = features.view(B, H * W, C)
    print(f"Token   shape : {tokens.shape}  (B, num_tokens, C)  <- feed this to your decoder")
    assert tokens.shape == (batch_size, 49, 256)

    print("Test passed.")


if __name__ == "__main__":
    test_feature_extraction()
