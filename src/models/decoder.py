import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from mamba_ssm import Mamba
except ImportError:
    class Mamba:
        def __init__(self, *_, **_kw):
            del _kw
            raise ImportError("mamba_ssm is required for MambaDecoder. Install with: pip install mamba-ssm --no-build-isolation")

try:
    from mamba_ssm import Mamba3
except ImportError:
    class Mamba3:
        def __init__(self, *_, **_kw):
            del _kw
            raise ImportError("mamba_ssm>=2.3 with Mamba3 is required for Mamba3Decoder. Install with: pip install mamba-ssm --no-build-isolation")

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
        max_len=100,
        use_checkpoint=False,
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
        self.use_checkpoint = use_checkpoint

    def forward(self, tgt, global_feat, memory):

        x = self.embedding(tgt)

        x = x + self.pos_embedding[:, :x.size(1)]

        tgt_mask = self.generate_mask(x.size(1)).to(x.device)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, global_feat, memory, tgt_mask, use_reentrant=False
                )
            else:
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

class Mamba3DecoderBlock(nn.Module):
    """Mamba3 SSM + cross-attention with pre-norm residual connections.

    Replaces the prior raw-Mamba3 stack: each block now has explicit residual
    norms between sub-layers and a cross-attention path that lets text tokens
    attend directly to the visual memory instead of relying on prefix
    concatenation through a causal SSM.
    """
    def __init__(self, d_model=512, num_heads=8, d_state=64, headdim=128,
                 is_mimo=False, mimo_rank=4, chunk_size=16,
                 dropout=0.1, dtype=torch.bfloat16):
        super().__init__()
        self.norm_ssm = nn.LayerNorm(d_model, dtype=dtype)
        self.mamba = Mamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            is_outproj_norm=True,
            dtype=dtype,
        )

        self.norm_cross = nn.LayerNorm(d_model, dtype=dtype)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True, dtype=dtype
        )

    def forward(self, x, memory):
        x = x + self.mamba(self.norm_ssm(x))
        h = self.norm_cross(x)
        attn_out, _ = self.cross_attn(h, memory, memory)
        x = x + attn_out
        return x


class Mamba3DecoderBlock(nn.Module):
    """Hybrid Mamba + Cross-Attention Block"""
    def __init__(self, d_model=512, num_heads=8, d_state=64, headdim=128,
                 is_mimo=False, mimo_rank=4, chunk_size=16,
                 dropout=0.1, dtype=torch.bfloat16):
        super().__init__()
        self.norm_ssm = nn.LayerNorm(d_model, dtype=dtype)
        self.mamba = Mamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            is_outproj_norm=True,
            dtype=dtype,
        )

        self.norm_cross = nn.LayerNorm(d_model, dtype=dtype)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True, dtype=dtype
        )

    def forward(self, x, memory, pad_mask=None):
        # 1. SSM Text Modeling
        residual = x
        x = self.norm_ssm(x)
        
        # STOP THE HANG: Zero out the padded regions BEFORE passing to Mamba
        if pad_mask is not None:
            x = x * pad_mask 
            
        x = residual + self.mamba(x)
        
        # 2. Cross-Attention to Image Memory (High CIDEr mechanism)
        h = self.norm_cross(x)
        attn_out, _ = self.cross_attn(h, memory, memory)
        x = x + attn_out
        
        return x


class Mamba3Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        encoder_dim=768,       
        d_model=512,           
        num_heads=8,           
        d_state=64,            
        headdim=128,           
        num_layers=6,          
        is_mimo=False,         
        mimo_rank=4,
        chunk_size=16,
        max_len=128,
        dropout=0.1,
        dtype=torch.bfloat16,  
        pad_token_id=0         
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.dtype = dtype

        self.visual_proj = nn.Linear(encoder_dim, d_model, dtype=dtype)
        self.visual_norm = nn.LayerNorm(d_model, dtype=dtype)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id, dtype=dtype)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, d_model, dtype=dtype) * 0.02
        )

        self.layers = nn.ModuleList([
            Mamba3DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_state=d_state,
                headdim=headdim,
                is_mimo=is_mimo,
                mimo_rank=mimo_rank,
                chunk_size=chunk_size,
                dropout=dropout,
                dtype=dtype,
            ) for _ in range(num_layers)
        ])

        self.norm_f = nn.LayerNorm(d_model, dtype=dtype)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, dtype=dtype)

        self.lm_head.weight = self.embedding.weight
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        # STOP THE HANG: Explicitly zero the padding embedding on init
        with torch.no_grad():
            self.embedding.weight[self.pad_token_id].fill_(0.0)

    def forward(self, image_features, text_input_ids):
        if image_features.dim() == 4:
            B, H, W, C = image_features.shape
            image_features = image_features.view(B, H * W, C)

        # Process Visual Features (Memory)
        image_features = image_features.to(self.dtype)
        memory = self.visual_norm(self.visual_proj(image_features))  

        # Process Text 
        text_embeds = self.embedding(text_input_ids)
        text_embeds = text_embeds + self.pos_embedding[:, :text_embeds.size(1)]

        # Create binary mask to prevent Mamba NaN explosions (1 for text, 0 for pad)
        pad_mask = (text_input_ids != self.pad_token_id).unsqueeze(-1).to(self.dtype)

        hidden_states = text_embeds.to(device=image_features.device, dtype=self.dtype).contiguous()
        
        # Pass through Hybrid Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, memory, pad_mask=pad_mask)

        hidden_states = self.norm_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits.to(torch.float32)

    @torch.no_grad()
    def generate(self, image_features, start_token_id, end_token_id, max_length=20):
        self.eval()
        B = image_features.size(0)
        
        generated_ids = torch.full((B, 1), start_token_id, dtype=torch.long, device=image_features.device)

        for _ in range(max_length):
            logits = self.forward(image_features, generated_ids)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            if (generated_ids == end_token_id).any(dim=1).all():
                break
                
        return generated_ids