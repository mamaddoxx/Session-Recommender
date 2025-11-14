import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi

# -------------------------------
# CONFIGURATION (LOCAL MACHINE)
# -------------------------------

MAX_SESSIONS = 200_000
MIN_SESSION_LEN = 2
MAX_SESSION_LEN = 50

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_FILE = RAW_DIR / "yoochoose-clicks.dat"

KAGGLE_DATASET = "chadgostopp/recsys-challenge-2015"
KAGGLE_FILENAME = "yoochoose-clicks.dat"


# -------------------------------
# STEP 1 — Download dataset
# -------------------------------

def download_dataset():
    print("[INFO] Downloading RecSys 2015 dataset from Kaggle...")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(RAW_DIR),
        unzip=True
    )

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"[ERROR] After downloading, {RAW_FILE} is missing."
        )

    print(f"[OK] Dataset downloaded to: {RAW_FILE}")


# -------------------------------
# STEP 2 — Load raw dataset
# -------------------------------

def load_raw_clicks():
    print(f"[INFO] Loading raw clickstream from {RAW_FILE}")

    cols = ["session_id", "timestamp", "item_id", "category"]

    df = pd.read_csv(
        RAW_FILE,
        sep=",",
        header=None,
        names=cols
    )

    return df


# -------------------------------
# STEP 3 — Build sessions
# -------------------------------

def build_sessions(df):

    df = df.sort_values(["session_id", "timestamp"])

    unique = df["session_id"].unique()[:MAX_SESSIONS]
    df = df[df["session_id"].isin(unique)]

    sessions = []
    for sid, g in df.groupby("session_id"):
        items = g["item_id"].tolist()
        if len(items) >= MIN_SESSION_LEN:
            sessions.append(items)

    print(f"[INFO] Built {len(sessions)} sessions")
    return sessions


# -------------------------------
# STEP 4 — Build mapping
# -------------------------------

def build_item_mapping(sessions):
    items = sorted({x for seq in sessions for x in seq})

    item2idx = {iid: idx+1 for idx, iid in enumerate(items)}
    idx2item = {idx+1: iid for idx, iid in enumerate(items)}

    print(f"[INFO] Found {len(items)} unique items")
    return item2idx, idx2item


# -------------------------------
# STEP 5 — Encode & Split
# -------------------------------

def encode_and_split(sessions, item2idx):

    X, y, L = [], [], []

    for seq in sessions:
        if len(seq) > MAX_SESSION_LEN:
            seq = seq[-MAX_SESSION_LEN:]

        X.append([item2idx[i] for i in seq[:-1]])
        y.append(item2idx[seq[-1]])
        L.append(len(seq)-1)

    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    def pick(indices):
        return [X[i] for i in indices], [y[i] for i in indices], [L[i] for i in indices]

    return pick(train_idx), pick(val_idx), pick(test_idx)


# -------------------------------
# STEP 6 — Pad sequences
# -------------------------------

def pad(sequences):
    padded = []
    lengths = []

    for seq in sequences:
        l = len(seq)
        if l > MAX_SESSION_LEN:
            seq = seq[-MAX_SESSION_LEN:]
            l = MAX_SESSION_LEN

        padded.append([0]*(MAX_SESSION_LEN-l) + seq)
        lengths.append(l)

    return np.array(padded), np.array(lengths)


# -------------------------------
# MAIN PIPELINE
# -------------------------------

def main():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    download_dataset()
    df = load_raw_clicks()
    sessions = build_sessions(df)
    item2idx, idx2item = build_item_mapping(sessions)

    (X_train, y_train, L_train),(X_val, y_val, L_val),(X_test, y_test, L_test) = encode_and_split(sessions, item2idx)

    X_train_pad, L_train = pad(X_train)
    X_val_pad, L_val = pad(X_val)
    X_test_pad, L_test = pad(X_test)

    np.save(PROCESSED_DIR / "X_train.npy", X_train_pad)
    np.save(PROCESSED_DIR / "y_train.npy", np.array(y_train))
    np.save(PROCESSED_DIR / "L_train.npy", L_train)

    np.save(PROCESSED_DIR / "X_val.npy", X_val_pad)
    np.save(PROCESSED_DIR / "y_val.npy", np.array(y_val))
    np.save(PROCESSED_DIR / "L_val.npy", L_val)

    np.save(PROCESSED_DIR / "X_test.npy", X_test_pad)
    np.save(PROCESSED_DIR / "y_test.npy", np.array(y_test))
    np.save(PROCESSED_DIR / "L_test.npy", L_test)

    np.save(PROCESSED_DIR / "item2idx.npy", item2idx)
    np.save(PROCESSED_DIR / "idx2item.npy", idx2item)

    print(f"[DONE] Saved processed files to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
