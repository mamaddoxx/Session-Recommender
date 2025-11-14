# src/train_sasrec.py

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.recsys_models import create_dataloader
from src.models.sasrec import SASRec


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def recall_mrr_at_k(model, dataloader, ks=(10, 20), device=DEVICE):
    model.eval()
    ks = sorted(ks)
    recalls = {k: 0 for k in ks}
    mrrs = {k: 0.0 for k in ks}
    n = 0

    with torch.no_grad():
        for x, y, lengths in dataloader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            logits = model(x, lengths)             # (B, num_items+1)
            max_k = max(ks)
            _, ranked = torch.topk(logits, k=max_k, dim=1)

            for i in range(x.size(0)):
                true_item = y[i].item()
                preds = ranked[i].tolist()
                n += 1
                for k in ks:
                    topk = preds[:k]
                    if true_item in topk:
                        recalls[k] += 1
                        rank = topk.index(true_item) + 1
                        mrrs[k] += 1.0 / rank

    recalls = {k: recalls[k] / n for k in ks}
    mrrs = {k: mrrs[k] / n for k in ks}
    return recalls, mrrs


def train_sasrec(
    data_dir: Path,
    model_dir: Path,
    batch_size: int = 256,
    num_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 512,
    dropout: float = 0.2,
):

    # ----- Load data -----
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    L_train = np.load(data_dir / "L_train.npy")

    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    L_val = np.load(data_dir / "L_val.npy")

    item2idx = np.load(data_dir / "item2idx.npy", allow_pickle=True).item()
    num_items = len(item2idx)
    max_seq_len = X_train.shape[1]

    print(f"Num items: {num_items}, max_seq_len: {max_seq_len}")

    train_loader = create_dataloader(X_train, y_train, L_train, batch_size=batch_size, shuffle=True)
    val_loader   = create_dataloader(X_val,   y_val,   L_val,   batch_size=batch_size, shuffle=False)

    # ----- Build model -----
    model = SASRec(
        num_items=num_items,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        padding_idx=0,
        tie_weights=True,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    model_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"[SASRec] Epoch {epoch}/{num_epochs}")
        for x, y, lengths in pbar:
            x, y, lengths = x.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x, lengths)        # (B, num_items+1)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_count += bs

            pbar.set_postfix({"loss": total_loss / total_count})

        train_loss = total_loss / total_count

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for x, y, lengths in val_loader:
                x, y, lengths = x.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)
                logits = model(x, lengths)
                loss = criterion(logits, y)
                bs = x.size(0)
                val_loss += loss.item() * bs
                val_count += bs

        val_loss /= val_count
        recalls, mrrs = recall_mrr_at_k(model, val_loader, ks=(10, 20), device=DEVICE)

        print(
            f"[SASRec] Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"R@10={recalls[10]:.4f}, R@20={recalls[20]:.4f}, "
            f"MRR@10={mrrs[10]:.4f}, MRR@20={mrrs[20]:.4f}"
        )

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_dir / "sasrec_best.pt")
            print("  -> Saved new best SASRec model")

        scheduler.step()

    print("Training finished.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    train_sasrec(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
