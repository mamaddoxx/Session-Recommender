# src/models/sasrec.py

import torch
from torch import nn
import torch.nn.functional as F


class SASRecBlock(nn.Module):
    """
    One transformer block used in SASRec:
    - Multi-head self-attention (causal)
    - Residual + LayerNorm
    - Position-wise FFN + residual + LayerNorm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # (B, T, D)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, T, D)
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,               # (T, T) causal mask
            key_padding_mask=key_padding_mask  # (B, T) padding mask
        )
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class SASRec(nn.Module):
    """
    Production-style SASRec for next-item prediction.

    - item_emb: shared input & (optional) output weights
    - pos_emb: learnable positional embeddings
    - N transformer blocks with MHSA + FFN
    - Causal attention mask + padding mask
    - Predicts next item from the last non-padded position

    Inputs:
        x:       (B, T)  padded item indices, 0 = PAD
        lengths: (B,)    true sequence lengths
    Returns:
        logits:  (B, num_items+1)  scores for next item
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.2,
        padding_idx: int = 0,
        tie_weights: bool = True,
    ):
        super().__init__()

        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.padding_idx = padding_idx

        # ----- Embeddings -----
        self.item_emb = nn.Embedding(
            num_embeddings=num_items + 1,      # +1 for PAD
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=d_model,
        )

        # ----- Transformer blocks -----
        self.blocks = nn.ModuleList([
            SASRecBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # ----- Output projection -----
        self.out = nn.Linear(d_model, num_items + 1, bias=False)

        if tie_weights:
            # tie output weights with item embeddings (common trick)
            self.out.weight = self.item_emb.weight

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        for block in self.blocks:
            for m in block.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _generate_causal_mask(self, T: int, device):
        # True where attention is NOT allowed (future positions)
        # Shape: (T, T)
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return mask  # used as attn_mask

    def forward(self, x, lengths):
        """
        x:       (B, T)  padded sequences
        lengths: (B,)    true lengths
        """
        device = x.device
        B, T = x.size()

        # ----- Embedding + position -----
        item_emb = self.item_emb(x)  # (B, T, D)

        # position indices: 0..T-1 for each batch
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_emb(pos_ids)  # (B, T, D)

        h = item_emb + pos_emb
        h = self.dropout(h)

        # ----- Masks -----
        causal_mask = self._generate_causal_mask(T, device=device)  # (T, T)
        # True where padding (should NOT be attended to)
        key_padding_mask = (x == self.padding_idx)  # (B, T)

        # ----- Transformer blocks -----
        for block in self.blocks:
            h = block(
                h,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
            )

        h = self.layer_norm(h)  # (B, T, D)

        # ----- Take last non-padded position for each sequence -----
        # lengths are 1-based, so last index = lengths - 1
        last_idx = (lengths - 1).clamp(min=0)  # (B,)
        batch_idx = torch.arange(B, device=device)
        last_hidden = h[batch_idx, last_idx]   # (B, D)

        # ----- Project to item logits -----
        logits = self.out(last_hidden)  # (B, num_items+1)
        return logits
