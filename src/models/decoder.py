import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class PureTDecoderBlock(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()

        self.prefusion = nn.Linear(dim * 2, dim)
        self.norm_prefusion = nn.LayerNorm(dim)

        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, global_feat, memory, tgt_mask=None):

        global_expand = global_feat.unsqueeze(1).expand(-1, x.size(1), -1)

        x_prefuse = torch.cat([x, global_expand], dim=-1)
        x_prefuse = self.prefusion(x_prefuse)

        x = self.norm_prefusion(x + x_prefuse)

        attn_out, _ = self.self_attn(
            x, x, x, attn_mask=tgt_mask
        )

        x = self.norm1(x + attn_out)

        cross_out, _ = self.cross_attn(
            x, memory, memory
        )

        x = self.norm2(x + cross_out)

        ff_out = self.ff(x)

        x = self.norm3(x + ff_out)

        return x
    
class PureTDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=512,
        num_heads=8,
        num_layers=3,
        max_len=100
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, dim)
        )

        self.layers = nn.ModuleList([
            PureTDecoderBlock(dim, num_heads)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, tgt, global_feat, memory):

        x = self.embedding(tgt)

        x = x + self.pos_embedding[:, :x.size(1)]

        tgt_mask = self.generate_mask(x.size(1)).to(x.device)

        for layer in self.layers:
            x = layer(x, global_feat, memory, tgt_mask)

        logits = self.fc(x)

        return logits

    def generate_mask(self, size):
        mask = torch.triu(
            torch.ones(size, size), diagonal=1
        ).bool()
        return mask

class MambaDecoderBlock(nn.Module):
    def __init__(self, dim=512):
        super().__init__()

        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.mamba(x)
        x = self.norm(x + residual)
        return x
    
class MambaDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=512,
        num_layers=6,
        max_len=128
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, dim)
        )

        self.layers = nn.ModuleList([
            MambaDecoderBlock(dim)
            for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, tokens, visual_features=None):

        x = self.embedding(tokens)

        x = x + self.pos_embedding[:, :x.size(1)]

        if visual_features is not None:
            x = x + visual_features.unsqueeze(1)

        for layer in self.layers:
            x = layer(x)

        logits = self.lm_head(x)

        return logits