import torch
import torch.nn as nn
import timm


# ─────────────────────────────────────────────
# 1. Patch Embedding
# ─────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and project each to embed_dim.
    Image: (B, C, H, W)  →  Patches: (B, num_patches, embed_dim)
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."

        self.num_patches = (img_size // patch_size) ** 2

        # Conv2d with kernel=patch_size and stride=patch_size is equivalent
        # to splitting the image into patches and linearly projecting each one
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)          # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)          # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, embed_dim)
        return x


# ─────────────────────────────────────────────
# 2. Multi-Head Self-Attention
# ─────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)                              # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)              # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)                        # each: (B, heads, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate values
        x = (attn @ v)                                 # (B, heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)        # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ─────────────────────────────────────────────
# 3. MLP (Feed-Forward Block)
# ─────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, in_features),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 4. Transformer Encoder Block
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, int(embed_dim * mlp_ratio), drop=proj_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # Pre-norm + residual
        x = x + self.mlp(self.norm2(x))    # Pre-norm + residual
        return x


# ─────────────────────────────────────────────
# 5. Vision Transformer (no classification head)
# ─────────────────────────────────────────────
class ViT(nn.Module):
    """
    Vision Transformer backbone — no classification head.

    Returns either:
      - CLS token:         (B, embed_dim)          when return_all_tokens=False
      - All token states:  (B, num_patches+1, embed_dim)  when return_all_tokens=True

    Args:
        img_size        : input image resolution (default 224)
        patch_size      : patch size in pixels   (default 16)
        in_channels     : number of input channels (default 3)
        embed_dim       : token embedding dimension (default 768)
        depth           : number of Transformer blocks (default 12)
        num_heads       : attention heads per block  (default 12)
        mlp_ratio       : MLP hidden dim = embed_dim * mlp_ratio (default 4)
        attn_drop       : dropout on attention weights (default 0.0)
        proj_drop       : dropout on projections / MLP (default 0.0)
        return_all_tokens: if True return all token states, else CLS only
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        return_all_tokens=False,
    ):
        super().__init__()
        self.return_all_tokens = return_all_tokens

        # ── Patch embedding ──────────────────────────────────
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # ── Learnable tokens & positional encoding ───────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(proj_drop)

        # ── Transformer encoder ──────────────────────────────
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, attn_drop, proj_drop)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # ── Weight initialisation ────────────────────────────
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        # 1. Patch embedding
        x = self.patch_embed(x)                        # (B, num_patches, embed_dim)

        # 2. Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)                 # (B, num_patches+1, embed_dim)

        # 3. Add positional encoding
        x = self.pos_drop(x + self.pos_embed)

        # 4. Transformer blocks
        x = self.blocks(x)                             # (B, num_patches+1, embed_dim)
        x = self.norm(x)

        # 5. Return CLS token or all tokens
        if self.return_all_tokens:
            return x                                   # (B, num_patches+1, embed_dim)
        return x[:, 0]                                 # (B, embed_dim)  ← CLS token only


# ─────────────────────────────────────────────
# 6. Convenience constructors (ViT-B / ViT-L)
# ─────────────────────────────────────────────
def vit_base(patch_size=16, **kwargs):
    return ViT(patch_size=patch_size, embed_dim=768,  depth=12, num_heads=12, **kwargs)

def vit_large(patch_size=16, **kwargs):
    return ViT(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, **kwargs)


# ─────────────────────────────────────────────
# 7. Pretrained ViT via timm
# ─────────────────────────────────────────────
class PretrainedViT(nn.Module):
    """
    Thin wrapper around a timm pretrained ViT backbone.

    Returns either:
      - All patch tokens : (B, num_patches, embed_dim)  when return_patch_tokens=True  (default)
      - CLS token only   : (B, embed_dim)               when return_patch_tokens=False
    """
    def __init__(self, model_name: str, pretrained: bool = True, return_patch_tokens: bool = True):
        super().__init__()
        # num_classes=0 removes the classification head; global_pool='' keeps all tokens
        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                          num_classes=0, global_pool="")
        self.return_patch_tokens = return_patch_tokens
        self.embed_dim = self.backbone.embed_dim

    def forward(self, x):
        # timm forward_features returns (B, num_patches+1, embed_dim) — CLS at index 0
        tokens = self.backbone(x)          # (B, num_patches+1, embed_dim)
        if self.return_patch_tokens:
            return tokens[:, 1:, :]        # (B, num_patches, embed_dim)
        return tokens[:, 0]               # (B, embed_dim)


def vit_base_pretrained(return_patch_tokens: bool = True) -> PretrainedViT:
    """ViT-B/16 pretrained on ImageNet-21k → embed_dim=768, num_patches=196."""
    return PretrainedViT("vit_base_patch16_224", pretrained=True,
                         return_patch_tokens=return_patch_tokens)


def vit_small_pretrained(return_patch_tokens: bool = True) -> PretrainedViT:
    """ViT-S/16 pretrained on ImageNet-1k → embed_dim=384, num_patches=196."""
    return PretrainedViT("vit_small_patch16_224", pretrained=True,
                         return_patch_tokens=return_patch_tokens)


