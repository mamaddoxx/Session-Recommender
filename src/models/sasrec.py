# src/models/sasrec.py

import torch
from torch import nn


class SASRec(nn.Module):
    """
    Simplified, numerically-stable SASRec-style model using TransformerEncoder.

    - No manual causal mask (avoids NaN issues).
    - Uses padding mask so PAD tokens are ignored.
    - Still uses positional + item embeddings and Transformer blocks.
    - Next-item prediction from the last non-padded position.

    Inputs:
        x:       (B, T)  padded item indices, 0 = PAD
        lengths: (B,)    true sequence lengths

    Output:
        logits:  (B, num_items+1) scores for next item
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
            num_embeddings=num_items + 1,      # +1 for PAD index
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=d_model,
        )

        # ----- Transformer encoder (no custom mask, just padding mask) -----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,  # (B, T, D)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # ----- Output projection -----
        self.out = nn.Linear(d_model, num_items + 1, bias=False)

        if tie_weights:
            # Tie output weights with item embeddings
            self.out.weight = self.item_emb.weight

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, lengths):
        """
        x:       (B, T)
        lengths: (B,)
        """
        device = x.device
        B, T = x.size()

        # --- Embeddings ---
        item_emb = self.item_emb(x)  # (B, T, D)

        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_emb(pos_ids)  # (B, T, D)

        h = item_emb + pos_emb
        h = self.dropout(h)

        # --- Padding mask: True where we want to IGNORE (PAD positions) ---
        # nn.TransformerEncoder expects src_key_padding_mask with shape (B, T)
        # where True => position is ignored.
        key_padding_mask = (x == self.padding_idx)  # (B, T), bool

        # --- Transformer encoder ---
        # No attn_mask (no causal mask) -> full self-attention over sequence.
        h = self.encoder(
            h,
            src_key_padding_mask=key_padding_mask,
        )  # (B, T, D)

        h = self.layer_norm(h)

        # --- Take last non-padded position for each sequence ---
        last_idx = (lengths - 1).clamp(min=0)          # (B,)
        batch_idx = torch.arange(B, device=device)     # (B,)
        last_hidden = h[batch_idx, last_idx]           # (B, D)

        # --- Project to logits over items ---
        logits = self.out(last_hidden)  # (B, num_items+1)

        # Extra safety: replace any NaNs/Infs with finite numbers
        logits = torch.nan_to_num(logits, neginf=-1e4, posinf=1e4)

        return logits
