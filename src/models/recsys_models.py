# src/models/recsys_models.py

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils


class SessionDataset(Dataset):
    def __init__(self, X, y, lengths):
        # X: (N, T), y: (N,), lengths: (N,)
        self.X = torch.from_numpy(X).long()
        self.y = torch.from_numpy(y).long()
        self.lengths = torch.from_numpy(lengths).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]


def create_dataloader(X, y, lengths, batch_size=256, shuffle=True):
    ds = SessionDataset(X, y, lengths)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class GRU4Rec(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        padding_idx: int = 0,
    ):
        """
        num_items: number of distinct items (max index in item2idx)
        """
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # +1 because index 0 is reserved for PAD
        self.item_emb = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, num_items + 1)

    def forward(self, x, lengths):
        """
        x: (B, T)      → padded item indices
        lengths: (B,)  → true sequence lengths (before padding)

        returns:
            logits: (B, num_items+1)
        """
        # (B, T, D)
        emb = self.item_emb(x)

        # pack padded sequence for GRU
        packed = rnn_utils.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # h_n: (num_layers, B, H)
        _, h_n = self.gru(packed)

        # take LAST layer's hidden state: (B, H)
        h = h_n[-1]  # shape (B, H)

        # final logits: (B, num_items+1)
        logits = self.fc(h)
        return logits
