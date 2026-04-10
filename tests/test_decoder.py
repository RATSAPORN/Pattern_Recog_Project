import sys
import os
import torch

# Ensure the src directory is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.decoder import Mamba3Decoder

def test_mamba3_decoder():
    print("--- Starting Mamba3Decoder Tests ---")
    
    # 1. Define Dummy Dimensions (Matching vanilla_vmamba_small output and Flickr8k EDA)
    batch_size = 2
    H, W = 7, 7                # Resulting feature map spatial dims
    num_patches = H * W        # 49 patches
    encoder_dim = 256          # Matched to your VSSM tiny config test output
    vocab_size = 2991          # Using Flickr8k reduced vocab size (freq >= 5)
    seq_len = 15               # Arbitrary text sequence length
    d_model = 256              # Mamba internal dimension

    # 2. Instantiate the Decoder
    # Using a tiny config for fast CPU/GPU testing
    model = Mamba3Decoder(
        vocab_size=vocab_size,
        encoder_dim=encoder_dim,
        d_model=d_model,
        d_state=32,            # Reduced for fast testing
        headdim=32,            # Reduced for fast testing
        num_layers=2,          # Reduced for fast testing
        is_mimo=True,
        mimo_rank=4,
        chunk_size=16,
        dtype=torch.bfloat16,
        pad_token_id=0
    )
    
    # Move to GPU if available, as Mamba optimizations often require CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 3. Create Dummy Inputs
    # Simulating VMamba output: (B, H, W, C)
    image_features = torch.randn(batch_size, H, W, encoder_dim, device=device, dtype=torch.float32)
    
    # Simulating tokenized ground truth captions: (B, seq_len)
    text_input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

    print(f"Input Image Features Shape : {image_features.shape}  (B, H, W, C)")
    print(f"Input Text Tokens Shape    : {text_input_ids.shape}       (B, seq_len)")

    # 4. Test Forward Pass (Training Mode Simulation)
    print("\n--- Testing Forward Pass (Teacher Forcing) ---")
    with torch.no_grad():
        logits = model(image_features, text_input_ids)
    
    print(f"Output Logits Shape        : {logits.shape}  (B, seq_len, vocab_size)")
    
    # Assertions for Forward Pass
    assert logits.shape == (batch_size, seq_len, vocab_size), "Logits shape mismatch!"
    # If you applied the float32 optimization, check dtype
    assert logits.dtype == torch.float32, f"Logits should be float32 for CrossEntropy, got {logits.dtype}"
    
    print("Forward pass successful.")

    # 5. Test Autoregressive Generation (Inference Mode Simulation)
    print("\n--- Testing Autoregressive Generation ---")
    start_token_id = 1 # Dummy <BOS> token
    end_token_id = 2   # Dummy <EOS> token
    max_gen_length = 20

    with torch.no_grad():
        generated_ids = model.generate(
            image_features=image_features,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            max_length=max_gen_length
        )
    
    print(f"Generated Sequence Shape   : {generated_ids.shape}  (B, generated_len)")
    
    # Assertions for Generation
    assert generated_ids.size(0) == batch_size, "Batch size mismatch in generation!"
    assert generated_ids.size(1) <= max_gen_length + 1, "Generated sequence exceeds max length!"
    assert (generated_ids[:, 0] == start_token_id).all(), "Sequence did not start with <BOS> token!"

    print("Generation pass successful.")
    print("\nAll Mamba3Decoder tests passed.")

if __name__ == "__main__":
    test_mamba3_decoder()