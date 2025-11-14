# src/models/recsys_models.py

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class SessionDataset(Dataset):
    def __init__(self, X, y, lengths):
        """
        X: numpy array (num_samples, max_len) of item indices
        y: numpy array (num_samples,) of target item indices
        lengths: numpy array (num_samples,) of sequence lengths
        """
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
        padding_idx: int = 0
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.item_emb = nn.Embedding(
            num_embeddings=num_items + 1,  # +1 because indices start from 1
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_items + 1)  # logits for all items

    def forward(self, x, lengths):
        """
        x: (B, T)
        lengths: (B,)
        """
        emb = self.item_emb(x)   # (B, T, D)

        # pack sequences (because of padding)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, h_n = self.gru(packed)  # h_n: (1, B, H)
        h = h_n.squeeze(0)         # (B, H)

        logits = self.fc(h)        # (B, num_items+1)
        return logits
