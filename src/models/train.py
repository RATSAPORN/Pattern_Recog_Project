"""
train.py

Image captioning: pretrained encoder + Transformer/Mamba decoder.
Evaluation: BLEU-1..4, ROUGE-L, METEOR, CIDEr via pycocoevalcap.

── Setup (RunPod / any Linux GPU machine) ────────────────────────────────────
    # 1. Download dataset
    python src/data/make_data.py

    # 2. Install extra deps
    pip install pycocoevalcap pycocotools timm

    # 3. Train on Flickr8k with default (ViT-Base + Transformer)
    python src/models/train.py --dataset flickr8k

    # Train with VMamba encoder
    python src/models/train.py --dataset flickr8k \
        --encoder vmamba_small --decoder transformer

    # Train with ViT-Small encoder (faster, less memory)
    python src/models/train.py --dataset flickr8k \
        --encoder vit_small --decoder transformer

    # 4. Resume training from last checkpoint
    python src/models/train.py --dataset flickr8k \
        --checkpoint models/last.pt

    # 5. Load best weights only (no resume, just inspect)
    python src/models/train.py --dataset flickr8k \
        --checkpoint models/best.pt --epochs 0
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import argparse
import time
from collections import Counter

from tqdm import tqdm

import torch
from torch.amp import GradScaler
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
from src.models.encoder_vit import vit_base_pretrained, vit_small_pretrained
from src.models.encoder_vmamba import vanilla_vmamba_small, vanilla_vmamba_small_fast, vanilla_vmamba_tiny, vanilla_vmamba_slim, vanilla_vmamba_slim_tiny
from src.models.decoder import PureTDecoder, MambaDecoder, Mamba3Decoder
from src.data.build_features import get_flickr8k_dataloaders, get_mscoco_dataloaders


# ─── Encoder / Decoder factories (shared with predict.py) ────────────────────
def build_encoder(name: str, use_checkpoint: bool = False):
    """Returns (encoder, encoder_dim)."""
    if name == "vit_base":
        enc = vit_base_pretrained(return_patch_tokens=True)
        return enc, enc.embed_dim                          # 768
    if name == "vit_small":
        enc = vit_small_pretrained(return_patch_tokens=True)
        return enc, enc.embed_dim                          # 384
    if name == "vmamba_small":
        return vanilla_vmamba_small(pretrained=True), 768
    if name == "vmamba_small_fast":
        return vanilla_vmamba_small_fast(pretrained=True), 768
    if name == "vmamba_tiny":
        return vanilla_vmamba_tiny(pretrained=True, use_checkpoint=use_checkpoint), 768
    if name == "vmamba_slim":
        return vanilla_vmamba_slim(pretrained=False), 768
    if name == "vmamba_slim_tiny":
        return vanilla_vmamba_slim_tiny(pretrained=False), 768
    raise ValueError(f"Unknown encoder: {name!r}. Choose: vit_base | vit_small | vmamba_small | vmamba_small_fast | vmamba_tiny | vmamba_slim | vmamba_slim_tiny")


def build_decoder(name: str, vocab_size: int, encoder_dim: int,
                  decoder_dim: int, num_layers: int, max_len: int):
    if name == "transformer":
        return PureTDecoder(vocab_size=vocab_size, dim=decoder_dim,
                            num_heads=8, num_layers=num_layers, max_len=max_len)
    if name == "mamba":
        return MambaDecoder(vocab_size=vocab_size, dim=decoder_dim,
                            num_layers=num_layers, max_len=max_len)
    if name == "mamba3":
        return Mamba3Decoder(vocab_size=vocab_size, encoder_dim=encoder_dim,
                             d_model=decoder_dim, num_layers=num_layers)
    raise ValueError(f"Unknown decoder: {name!r}. Choose: transformer | mamba | mamba3")

# ─── special tokens ───────────────────────────────────────────────────────────
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"


# ─── Vocabulary ───────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.w2i = {PAD: 0, SOS: 1, EOS: 2, UNK: 3}
        self.i2w = {v: k for k, v in self.w2i.items()}

    def build(self, captions: list[str]):
        counts: Counter = Counter()
        for cap in captions:
            counts.update(cap.lower().split())
        for word, freq in counts.items():
            if freq >= self.min_freq and word not in self.w2i:
                idx = len(self.w2i)
                self.w2i[word] = idx
                self.i2w[idx] = word

    def encode(self, caption: str, max_len: int) -> list[int]:
        tokens = [SOS] + caption.lower().split() + [EOS]
        tokens = tokens[:max_len]
        ids = [self.w2i.get(t, self.w2i[UNK]) for t in tokens]
        ids += [self.w2i[PAD]] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        words = []
        for i in ids:
            w = self.i2w.get(i, UNK)
            if w in (PAD, SOS):
                continue
            if w == EOS:
                break
            words.append(w)
        return " ".join(words)

    def __len__(self):
        return len(self.w2i)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.w2i, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        v = cls()
        with open(path) as f:
            v.w2i = json.load(f)
        v.i2w = {int(v2): k for k, v2 in v.w2i.items()}
        return v


# ─── Collate ──────────────────────────────────────────────────────────────────
def make_collate(vocab: Vocabulary, max_len: int):
    def collate(batch):
        images, captions = zip(*batch)
        images = torch.stack(images)
        ids = torch.tensor(
            [vocab.encode(c, max_len) for c in captions], dtype=torch.long
        )
        return images, ids
    return collate


# ─── Model wrapper ────────────────────────────────────────────────────────────
class CaptioningModel(nn.Module):
    """
    encoder  →  projection  →  decoder

    Supports any encoder/decoder combination via build_encoder() / build_decoder().
    """
    def __init__(self, encoder_name: str, decoder_name: str,
                 vocab_size: int, decoder_dim: int = 512,
                 num_layers: int = 3, max_len: int = 50,
                 freeze_encoder: bool = True, use_checkpoint: bool = False):
        super().__init__()
        self.encoder, encoder_dim = build_encoder(encoder_name, use_checkpoint=use_checkpoint)
        self.decoder = build_decoder(decoder_name, vocab_size, encoder_dim,
                                     decoder_dim, num_layers, max_len)
        self.proj = nn.Linear(encoder_dim, decoder_dim)
        self.decoder_name = decoder_name

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def _encode(self, images: torch.Tensor):
        feats = self.encoder(images)
        if feats.dim() == 4:                              # VMamba: (B,H,W,C)
            B, H, W, C = feats.shape
            feats = feats.view(B, H * W, C)
        memory      = self.proj(feats)                    # (B, tokens, decoder_dim)
        global_feat = memory.mean(dim=1)                  # (B, decoder_dim)
        return memory, global_feat

    def forward(self, images: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory, global_feat = self._encode(images)
        # Teacher forcing: input = tgt[:-1], target = tgt[1:]
        if self.decoder_name == "transformer":
            return self.decoder(tgt[:, :-1], global_feat, memory)
        if self.decoder_name == "mamba":
            return self.decoder(tgt[:, :-1], visual_features=global_feat)
        if self.decoder_name == "mamba3":
            return self.decoder(memory, tgt[:, :-1])
        raise ValueError(f"Unknown decoder: {self.decoder_name}")

    @torch.no_grad()
    def generate(self, images: torch.Tensor, vocab: Vocabulary,
                 max_len: int = 50, device: torch.device = torch.device("cpu")) -> list[str]:
        self.eval()
        memory, global_feat = self._encode(images)
        B = images.size(0)
        generated = torch.full((B, 1), vocab.w2i[SOS], dtype=torch.long, device=device)

        for _ in range(max_len):
            if self.decoder_name == "transformer":
                logits = self.decoder(generated, global_feat, memory)
            elif self.decoder_name == "mamba":
                logits = self.decoder(generated, visual_features=global_feat)
            elif self.decoder_name == "mamba3":
                logits = self.decoder(memory, generated)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)
            if (next_id == vocab.w2i[EOS]).all():
                break

        return [vocab.decode(row.tolist()) for row in generated]


# ─── Evaluation helpers ───────────────────────────────────────────────────────
def compute_metrics(hypotheses: dict, references: dict, bertscore: bool = False) -> dict:
    """
    Compute BLEU-1..4, ROUGE-L, METEOR, CIDEr using pycocoevalcap.
    Optionally compute BERTScore F1 (semantic similarity).

    Args:
        hypotheses : {image_id: [predicted_caption]}
        references : {image_id: [ref1, ref2, ...]}
        bertscore  : if True, also compute BERTScore F1 (requires bert-score package)
    Returns:
        dict of metric name → score
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.cider.cider import Cider

    scorers = [
        (Bleu(4),   ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Rouge(),   ["ROUGE-L"]),
        (Meteor(),  ["METEOR"]),
        (Cider(),   ["CIDEr"]),
    ]

    results = {}
    for scorer, names in scorers:
        score, _ = scorer.compute_score(references, hypotheses)
        if isinstance(score, list):
            for name, s in zip(names, score):
                results[name] = round(s, 4)
        else:
            results[names[0]] = round(score, 4)

    if bertscore:
        try:
            from bert_score import score as bert_score_fn
            # BERTScore needs flat lists; for multi-reference images use the first reference
            ids   = sorted(hypotheses.keys())
            # Use all references: repeat each prediction for every reference, then average
            refs_flat, preds_flat = [], []
            for i in ids:
                for ref in references[i]:
                    preds_flat.append(hypotheses[i][0])
                    refs_flat.append(ref)
            _, _, F1 = bert_score_fn(preds_flat, refs_flat,
                                     lang="en", verbose=False)
            # Average F1 per image (each image may have multiple refs)
            per_image_f1, idx = [], 0
            for i in ids:
                n = len(references[i])
                per_image_f1.append(F1[idx:idx + n].mean().item())
                idx += n
            results["BERTScore-F1"] = round(sum(per_image_f1) / len(per_image_f1), 4)
        except ImportError:
            print("BERTScore skipped — run: pip install bert-score")

    return results


# ─── Training / Validation ────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, vocab, grad_accum=1, scaler=None):
    model.train()
    model.encoder.eval()  # keep encoder in eval (frozen BN / dropout)
    total_loss, n_tokens = 0.0, 0

    optimizer.zero_grad()
    pbar = tqdm(enumerate(loader), total=len(loader), desc="  train", leave=False)
    for step, (images, captions) in pbar:
        images   = images.to(device)
        captions = captions.to(device)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits  = model(images, captions)
            targets = captions[:, 1:]
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            ) / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        mask = targets != vocab.w2i[PAD]
        total_loss += loss.item() * grad_accum * mask.sum().item()
        n_tokens   += mask.sum().item()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=f"{total_loss / max(n_tokens, 1):.4f}")

    return total_loss / max(n_tokens, 1)


@torch.no_grad()
def validate(model, loader, criterion, vocab, device, max_len: int = 50):
    model.eval()
    total_loss, n_tokens = 0.0, 0
    hypotheses, references = {}, {}
    img_idx = 0

    for images, captions in tqdm(loader, desc="  valid", leave=False):
        images   = images.to(device)
        captions = captions.to(device)

        # Loss
        logits  = model(images, captions)
        targets = captions[:, 1:]
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        mask = targets != vocab.w2i[PAD]
        total_loss += loss.item() * mask.sum().item()
        n_tokens   += mask.sum().item()

        # Greedy decode for metrics
        preds = model.generate(images, vocab, max_len=max_len, device=device)
        for i, pred in enumerate(preds):
            ref_ids  = captions[i].tolist()
            ref_text = vocab.decode(ref_ids)
            hypotheses[str(img_idx)] = [pred]
            references[str(img_idx)] = [ref_text]
            img_idx += 1

    val_loss = total_loss / max(n_tokens, 1)
    metrics  = compute_metrics(hypotheses, references)
    return val_loss, metrics


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder",     choices=["vit_base", "vit_small", "vmamba_small", "vmamba_small_fast", "vmamba_tiny", "vmamba_slim", "vmamba_slim_tiny"],
                        default="vit_base")
    parser.add_argument("--decoder",     choices=["transformer", "mamba", "mamba3"],
                        default="transformer")
    parser.add_argument("--dataset",     choices=["flickr8k", "mscoco"], default="flickr8k",
                        help="Which dataset to use (paths resolved from --data_dir)")
    parser.add_argument("--data_dir",    default="data",
                        help="Root data directory (created by make_data.py)")
    # Override individual paths if needed
    parser.add_argument("--image_dir",   default=None)
    parser.add_argument("--ann_json",    default=None)
    parser.add_argument("--train_split", default=None)
    parser.add_argument("--val_split",   default=None)
    parser.add_argument("--vocab_path",  default="models/vocab.json")
    parser.add_argument("--checkpoint",  default=None,
                        help="Path to .pt file to resume training or load weights")
    parser.add_argument("--save_dir",    default="models")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=4e-4)
    parser.add_argument("--decoder_dim", type=int,   default=512)
    parser.add_argument("--num_layers",  type=int,   default=3)
    parser.add_argument("--max_len",     type=int,   default=50)
    parser.add_argument("--min_freq",    type=int,   default=2)
    parser.add_argument("--workers",     type=int,   default=2)
    parser.add_argument("--freeze_encoder",  action="store_true", default=True)
    parser.add_argument("--grad_checkpoint", action="store_true", default=False,
                        help="Gradient checkpointing in VMamba encoder — reduces VRAM ~40%")
    parser.add_argument("--grad_accum",      type=int, default=1,
                        help="Accumulate gradients over N steps — simulates larger batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Device: {device}")

    # ── Vocabulary ──────────────────────────────────────────────────────────
    # Use build_features to get raw loaders (no tokenisation) just to read captions
    if args.dataset == "flickr8k":
        raw_train, raw_val = get_flickr8k_dataloaders(
            args.data_dir, batch_size=args.batch_size, num_workers=args.workers)
    else:
        raw_train = get_mscoco_dataloaders(
            args.data_dir, split="train", batch_size=args.batch_size, num_workers=args.workers)
        raw_val   = get_mscoco_dataloaders(
            args.data_dir, split="val",   batch_size=args.batch_size, num_workers=args.workers)

    if os.path.exists(args.vocab_path):
        print(f"Loading vocabulary from {args.vocab_path}")
        vocab = Vocabulary.load(args.vocab_path)
    else:
        print("Building vocabulary...")
        all_captions = [cap for _, cap in raw_train.dataset.samples]
        vocab = Vocabulary(min_freq=args.min_freq)
        vocab.build(all_captions)
        vocab.save(args.vocab_path)
        print(f"Vocabulary size: {len(vocab)}")

    # ── Datasets & Loaders with training transforms + tokenised collate ──────
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    collate = make_collate(vocab, args.max_len)

    raw_train.dataset.transform = train_tf
    raw_val.dataset.transform   = val_tf

    train_loader = DataLoader(raw_train.dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate, pin_memory=True)
    val_loader   = DataLoader(raw_val.dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, collate_fn=collate, pin_memory=True)

    # ── Model ───────────────────────────────────────────────────────────────
    print(f"Encoder: {args.encoder}  |  Decoder: {args.decoder}")
    model = CaptioningModel(
        encoder_name=args.encoder,
        decoder_name=args.decoder,
        vocab_size=len(vocab),
        decoder_dim=args.decoder_dim,
        num_layers=args.num_layers,
        max_len=args.max_len,
        freeze_encoder=args.freeze_encoder,
        use_checkpoint=args.grad_checkpoint,
    ).to(device)

    start_epoch = 0
    best_cider  = 0.0
    ckpt        = {}

    if args.checkpoint:
        try:
            import numpy._core.multiarray
            torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
        except Exception:
            pass
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_cider  = ckpt.get("cider", 0.0)
        print(f"Loaded weights from {args.checkpoint}")
        print(f"  epoch={start_epoch}  best_CIDEr={best_cider:.4f}")
        if "metrics" in ckpt:
            m = ckpt["metrics"]
            print(f"  BLEU-4={m.get('BLEU-4',0):.4f}  METEOR={m.get('METEOR',0):.4f}  CIDEr={m.get('CIDEr',0):.4f}")

    # ── Optimizer & Loss ────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.w2i[PAD])

    if args.checkpoint and "optimizer" in ckpt:
        optimizer.state_dict()  # init optimizer state before loading
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        print("  Optimizer and scheduler state restored.")

    # ── Mixed precision scaler (created once, persists across epochs) ────────
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, vocab, grad_accum=args.grad_accum, scaler=scaler)
        val_loss, metrics = validate(model, val_loader, criterion, vocab, device, args.max_len)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"BLEU-4={metrics.get('BLEU-4', 0):.4f} | "
            f"ROUGE-L={metrics.get('ROUGE-L', 0):.4f} | "
            f"METEOR={metrics.get('METEOR', 0):.4f} | "
            f"CIDEr={metrics.get('CIDEr', 0):.4f} | "
            f"{elapsed:.0f}s"
        )

        cider = metrics.get("CIDEr", 0.0)
        is_best = cider > best_cider
        best_cider = max(cider, best_cider)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cider": best_cider,
            "metrics": metrics,
            "vocab_size": len(vocab),
            "vocab_path": args.vocab_path,
            "args": vars(args),
        }
        last_path = os.path.join(args.save_dir, "last.pt")
        torch.save(ckpt, last_path)
        if is_best:
            best_path = os.path.join(args.save_dir, "best.pt")
            torch.save(ckpt, best_path)
            print(f"  ✓ New best CIDEr={best_cider:.4f}  saved → {best_path}")


if __name__ == "__main__":
    main()