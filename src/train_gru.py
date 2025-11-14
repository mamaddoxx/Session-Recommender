# src/train_gru.py

import numpy as np
import torch
from torch import nn, optim
from pathlib import Path
from tqdm import tqdm

from src.models.recsys_models import GRU4Rec, create_dataloader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    L_train = np.load(DATA_DIR / "L_train.npy")

    X_val = np.load(DATA_DIR / "X_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")
    L_val = np.load(DATA_DIR / "L_val.npy")

    item2idx = np.load(DATA_DIR / "item2idx.npy", allow_pickle=True).item()

    num_items = len(item2idx)

    return (X_train, y_train, L_train,
            X_val, y_val, L_val,
            num_items)

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x, y, lengths in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            lengths = lengths.to(DEVICE)

            logits = model(x, lengths)
            loss = criterion(logits, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / total_count

def train():
    (X_train, y_train, L_train,X_val, y_val, L_val,num_items) = load_data()

    train_loader = create_dataloader(X_train, y_train, L_train, batch_size=256, shuffle=True)
    val_loader   = create_dataloader(X_val, y_val, L_val, batch_size=256, shuffle=False)

    model = GRU4Rec(num_items=num_items,embedding_dim=128,hidden_dim=256,padding_idx=0)

    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for x, y, lengths in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            lengths = lengths.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x, lengths)  # (B, num_items+1)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            pbar.set_postfix({"loss": total_loss / total_count})

        train_loss = total_loss / total_count
        val_loss = evaluate_model(model, val_loader, criterion)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / "gru4rec_best.pt")
            print("Saved new best model.")

if __name__ == "__main__":
    train()
