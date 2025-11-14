# src/evaluate.py

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from src.models.recsys_models import GRU4Rec, create_dataloader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

def load_test_data():
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    L_test = np.load(DATA_DIR / "L_test.npy")

    item2idx = np.load(DATA_DIR / "item2idx.npy", allow_pickle=True).item()
    num_items = len(item2idx)

    return X_test, y_test, L_test, num_items

def recall_mrr_at_k(model, dataloader, ks=(10, 20)):
    model.eval()
    ks = sorted(ks)

    recalls = {k: 0 for k in ks}
    mrrs = {k: 0.0 for k in ks}
    n_samples = 0

    with torch.no_grad():
        for x, y, lengths in tqdm(dataloader, desc="Evaluating"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            lengths = lengths.to(DEVICE)

            logits = model(x, lengths)  # (B, num_items+1)
            # we don't care about PAD index 0 so it's fine
            # get rank of true item
            batch_size = x.size(0)

            # sort predictions
            _, ranked_items = torch.topk(logits, k=max(ks), dim=1)  # (B, max_k)

            for i in range(batch_size):
                true_item = y[i].item()
                preds = ranked_items[i].cpu().tolist()
                n_samples += 1

                for k in ks:
                    topk = preds[:k]
                    if true_item in topk:
                        recalls[k] += 1
                        rank = topk.index(true_item) + 1  # 1-based
                        mrrs[k] += 1.0 / rank

    recalls = {k: recalls[k] / n_samples for k in ks}
    mrrs = {k: mrrs[k] / n_samples for k in ks}
    return recalls, mrrs

def main():
    X_test, y_test, L_test, num_items = load_test_data()
    test_loader = create_dataloader(X_test, y_test, L_test, batch_size=512, shuffle=False)

    model = GRU4Rec(num_items=num_items, embedding_dim=128, hidden_dim=256, padding_idx=0)
    model.load_state_dict(torch.load(MODEL_DIR / "gru4rec_best.pt", map_location=DEVICE))
    model.to(DEVICE)

    recalls, mrrs = recall_mrr_at_k(model, test_loader, ks=(10, 20))
    print("Recall@K:", recalls)
    print("MRR@K:", mrrs)

if __name__ == "__main__":
    main()
