"""
predict.py

Load a trained captioning model and generate captions for images.

Supports any encoder + decoder combination:
  Encoders : vit_base  | vit_small  | vmamba_small
  Decoders : transformer | mamba | mamba3

Usage examples
──────────────
# Single image
python src/models/predict.py \
    --checkpoint models/best.pt \
    --encoder vit_base \
    --decoder transformer \
    --image path/to/image.jpg

# Folder of images
python src/models/predict.py \
    --checkpoint models/best.pt \
    --encoder vit_base \
    --decoder transformer \
    --image_dir path/to/images/

# Override vocab path (default: read from checkpoint)
python src/models/predict.py \
    --checkpoint models/best.pt \
    --vocab models/vocab.json \
    --image photo.jpg
"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from src.models.encoder_vit import vit_base_pretrained, vit_small_pretrained
from src.models.encoder_vmamba import vanilla_vmamba_small, vanilla_vmamba_small_fast, vanilla_vmamba_tiny, vanilla_vmamba_slim, vanilla_vmamba_slim_tiny
from src.models.decoder import PureTDecoder, MambaDecoder, Mamba3Decoder

# ─── Special tokens (must match training) ─────────────────────────────────────
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"


# ─── Vocabulary ───────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self):
        self.w2i: dict = {}
        self.i2w: dict = {}

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        v = cls()
        with open(path) as f:
            v.w2i = json.load(f)
        v.i2w = {int(idx): word for word, idx in v.w2i.items()}
        return v

    def decode(self, ids: list) -> str:
        words = []
        for i in ids:
            w = self.i2w.get(int(i), UNK)
            if w in (PAD, SOS):
                continue
            if w == EOS:
                break
            words.append(w)
        return " ".join(words)

    def __len__(self):
        return len(self.w2i)


# ─── Encoder factory ──────────────────────────────────────────────────────────
def build_encoder(name: str) -> tuple[nn.Module, int]:
    """Returns (encoder, encoder_dim)."""
    if name == "vit_base":
        enc = vit_base_pretrained(return_patch_tokens=True)   # (B, 196, 768)
        return enc, enc.embed_dim
    if name == "vit_small":
        enc = vit_small_pretrained(return_patch_tokens=True)  # (B, 196, 384)
        return enc, enc.embed_dim
    if name == "vmamba_small":
        return vanilla_vmamba_small(pretrained=False), 768
    if name == "vmamba_small_fast":
        return vanilla_vmamba_small_fast(pretrained=False), 768
    if name == "vmamba_tiny":
        return vanilla_vmamba_tiny(pretrained=False), 768
    if name == "vmamba_slim":
        return vanilla_vmamba_slim(pretrained=False), 768
    if name == "vmamba_slim_tiny":
        return vanilla_vmamba_slim_tiny(pretrained=False), 768
    raise ValueError(f"Unknown encoder: {name!r}. Choose: vit_base | vit_small | vmamba_small | vmamba_small_fast | vmamba_tiny | vmamba_slim | vmamba_slim_tiny")


# ─── Decoder factory ──────────────────────────────────────────────────────────
def build_decoder(name: str, vocab_size: int, encoder_dim: int,
                  decoder_dim: int = 512, num_layers: int = 3,
                  max_len: int = 50) -> nn.Module:
    if name == "transformer":
        return PureTDecoder(
            vocab_size=vocab_size,
            dim=decoder_dim,
            num_heads=8,
            num_layers=num_layers,
            max_len=max_len,
        )
    if name == "mamba":
        return MambaDecoder(
            vocab_size=vocab_size,
            dim=decoder_dim,
            num_layers=num_layers,
            max_len=max_len,
        )
    if name == "mamba3":
        return Mamba3Decoder(
            vocab_size=vocab_size,
            encoder_dim=encoder_dim,
            d_model=decoder_dim,
            num_layers=num_layers,
        )
    raise ValueError(f"Unknown decoder: {name!r}. Choose: transformer | mamba | mamba3")


# ─── Full captioning model ────────────────────────────────────────────────────
class CaptioningModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.proj = nn.Linear(encoder_dim, decoder_dim)

    def encode(self, images: torch.Tensor):
        """Returns (memory, global_feat) ready for any decoder."""
        feats = self.encoder(images)

        # VMamba returns (B, H, W, C) — flatten spatial dims
        if feats.dim() == 4:
            B, H, W, C = feats.shape
            feats = feats.view(B, H * W, C)

        memory     = self.proj(feats)          # (B, tokens, decoder_dim)
        global_feat = memory.mean(dim=1)       # (B, decoder_dim)
        return memory, global_feat

    @torch.no_grad()
    def generate(self, images: torch.Tensor, vocab: Vocabulary,
                 max_len: int = 50, device: torch.device = torch.device("cpu")) -> list:
        self.eval()
        memory, global_feat = self.encode(images)

        B = images.size(0)
        generated = torch.full((B, 1), vocab.w2i[SOS], dtype=torch.long, device=device)

        for _ in range(max_len):
            if isinstance(self.decoder, PureTDecoder):
                logits = self.decoder(generated, global_feat, memory)
            elif isinstance(self.decoder, MambaDecoder):
                logits = self.decoder(generated, visual_features=global_feat)
            elif isinstance(self.decoder, Mamba3Decoder):
                logits = self.decoder(memory, generated)
            else:
                raise TypeError(f"Unsupported decoder type: {type(self.decoder)}")

            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)

            if (next_id == vocab.w2i[EOS]).all():
                break

        return [vocab.decode(row.tolist()) for row in generated]


# ─── Image loading ────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return TRANSFORM(img).unsqueeze(0)   # (1, 3, 224, 224)


# ─── Load checkpoint ──────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, encoder_name: str, decoder_name: str,
               vocab_path: str | None, decoder_dim: int, num_layers: int,
               max_len: int, device: torch.device) -> tuple:
    """
    Build model from args, load weights from checkpoint, return (model, vocab).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if "metrics" in ckpt:
        m = ckpt["metrics"]
        print(f"  epoch={ckpt.get('epoch', '?')}  "
              f"BLEU-4={m.get('BLEU-4', 0):.4f}  "
              f"METEOR={m.get('METEOR', 0):.4f}  "
              f"CIDEr={m.get('CIDEr', 0):.4f}")

    # Resolve vocab
    if vocab_path is None:
        vocab_path = ckpt.get("vocab_path", "models/vocab.json")
    vocab = Vocabulary.load(vocab_path)
    print(f"Vocabulary: {len(vocab)} tokens  ({vocab_path})")

    # Build model
    encoder, encoder_dim = build_encoder(encoder_name)
    decoder = build_decoder(decoder_name, len(vocab), encoder_dim,
                            decoder_dim, num_layers, max_len)
    model = CaptioningModel(encoder, decoder, encoder_dim, decoder_dim).to(device)

    # Load weights
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  Missing keys  : {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    print("  Weights loaded.")

    model.eval()
    return model, vocab


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Image captioning inference")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved .pt file (models/best.pt)")
    parser.add_argument("--encoder",    default="vit_base",
                        choices=["vit_base", "vit_small", "vmamba_small", "vmamba_small_fast", "vmamba_tiny", "vmamba_slim", "vmamba_slim_tiny"])
    parser.add_argument("--decoder",    default="transformer",
                        choices=["transformer", "mamba", "mamba3"])
    parser.add_argument("--vocab",      default=None,
                        help="Override vocab.json path (default: read from checkpoint)")

    # Input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",     help="Single image file")
    group.add_argument("--image_dir", help="Directory of images")

    # Model hyperparams (should match training)
    parser.add_argument("--decoder_dim", type=int, default=512)
    parser.add_argument("--num_layers",  type=int, default=3)
    parser.add_argument("--max_len",     type=int, default=50)
    parser.add_argument("--output",      default=None,
                        help="Save predictions to a JSON file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, vocab = load_model(
        checkpoint_path=args.checkpoint,
        encoder_name=args.encoder,
        decoder_name=args.decoder,
        vocab_path=args.vocab,
        decoder_dim=args.decoder_dim,
        num_layers=args.num_layers,
        max_len=args.max_len,
        device=device,
    )

    # Collect image paths
    if args.image:
        image_paths = [args.image]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = [
            os.path.join(args.image_dir, f)
            for f in sorted(os.listdir(args.image_dir))
            if os.path.splitext(f)[1].lower() in exts
        ]
        print(f"Found {len(image_paths)} images in {args.image_dir}")

    # Generate captions
    results = {}
    for path in image_paths:
        image = load_image(path).to(device)
        caption = model.generate(image, vocab, max_len=args.max_len, device=device)[0]
        fname = os.path.basename(path)
        results[fname] = caption
        print(f"{fname}: {caption}")

    # Optionally save to JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} predictions → {args.output}")


if __name__ == "__main__":
    main()
