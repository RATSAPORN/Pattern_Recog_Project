import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from mamba_ssm import Mamba3

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

class Mamba3Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        encoder_dim=768,       # Default output dim of vanilla_vmamba_small
        d_model=768,           # Dimension of Mamba-3
        d_state=128,           # Mamba-3 state expansion
        headdim=64,            # Mamba-3 headdim
        num_layers=6,          # Number of Mamba-3 blocks
        is_mimo=True,          # Enable MIMO for complex tracking
        mimo_rank=4,
        chunk_size=16,
        dtype=torch.bfloat16,  # Mamba-3 is highly optimized for bf16
        pad_token_id=0         # Your tokenizer's padding token
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.dtype = dtype

        # 1. Visual Feature Projection
        # Maps the VMamba features to the Mamba-3 hidden dimension
        self.visual_proj = nn.Linear(encoder_dim, d_model, dtype=dtype)

        # 2. Text Embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id, dtype=dtype)

        # 3. The Mamba-3 Backbone (Stack of blocks)
        self.layers = nn.ModuleList([
            Mamba3(
                d_model=d_model,
                d_state=d_state,
                headdim=headdim,
                is_mimo=is_mimo,
                mimo_rank=mimo_rank,
                chunk_size=chunk_size,
                is_outproj_norm=False,
                dtype=dtype,
            ) for _ in range(num_layers)
        ])

        # 4. Final Layer Norm and Language Modeling Head
        self.norm_f = nn.LayerNorm(d_model, dtype=dtype)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, dtype=dtype)

        # Weight Tying: Share weights between embedding and output layer (improves training)
        self.lm_head.weight = self.embedding.weight

    def forward(self, image_features, text_input_ids):
        """
        Args:
            image_features: Tensor from VMamba. Can be (B, C, H, W) or (B, num_patches, C).
            text_input_ids: Tensor of shape (B, seq_len) containing caption token IDs.
        Returns:
            logits: Tensor of shape (B, seq_len, vocab_size) predicting the NEXT word.
        """
        # --- 1. Process Visual Features ---
        if image_features.dim() == 4:
            # VMamba outputs (B, H, W, C). We just need to flatten H and W.
            B, H, W, C = image_features.shape
            image_features = image_features.view(B, H * W, C)
        
        # Cast to bf16 and project
        image_features = image_features.to(self.dtype)
        vis_embeds = self.visual_proj(image_features) # (B, num_patches, d_model)

        # --- 2. Process Text Features ---
        text_embeds = self.embedding(text_input_ids)  # (B, seq_len, d_model)

        # --- 3. Concatenate (Visual Prompting / Prefixing) ---
        # We prepend the visual tokens to the text tokens.
        # Shape becomes: (B, num_patches + seq_len, d_model)
        hidden_states = torch.cat([vis_embeds, text_embeds], dim=1)

        # --- 4. Pass through Mamba-3 Layers ---
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm_f(hidden_states)

        # --- 5. Extract Text Logits ---
        # Mamba-3 processed everything sequentially. 
        # We only care about the outputs corresponding to the text tokens to compute CrossEntropyLoss.
        num_patches = vis_embeds.size(1)
        text_hidden_states = hidden_states[:, num_patches:, :] 

        # Compute vocabulary logits
        logits = self.lm_head(text_hidden_states) # (B, seq_len, vocab_size)

        return logits.to(torch.float32) # Cast to float32 for stable loss calculation

    @torch.no_grad()
    def generate(self, image_features, start_token_id, end_token_id, max_length=20):
        """
        Basic greedy autoregressive generation loop for inference.
        Note: For maximum speed in production, you would utilize Mamba's KV-state caching, 
        but this loop works perfectly for validation and testing.
        """
        self.eval()
        B = image_features.size(0)
        
        # Start with the <BOS> token
        generated_ids = torch.full((B, 1), start_token_id, dtype=torch.long, device=image_features.device)

        for _ in range(max_length):
            # Forward pass with the sequence generated so far
            logits = self.forward(image_features, generated_ids)
            
            # Get the predicted token (last token in the sequence)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            # Stop early if all batches generated the <EOS> token
            if (generated_ids == end_token_id).any(dim=1).all():
                break
                
        return generated_ids