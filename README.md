#  Session-Based Sequential Recommender System

### GRU4Rec → Transformer (SASRec) Upgrade with Major Performance Improvements

This project implements an end-to-end **Session-Based Recommendation
System** using the **YooChoose** dataset.
It follows a full ML lifecycle: data preprocessing, EDA, baseline
modeling, Transformer modeling, evaluation, and production-level
engineering.

The system evolves from **GRU-based sequence models** to a modern
**Transformer-based SASRec**, achieving significant improvements in
Recall@20 and MRR@20.

------------------------------------------------------------------------

##  Project Highlights

 Full sequential modeling pipeline
 - Custom preprocessing and masking logic
 - Three GRU baselines (small/medium/large)
 - Full SASRec Transformer implementation
 - Stable training after attention-mask fixes


------------------------------------------------------------------------

##  Dataset

**YooChoose Clicks** dataset
- \~Milions click events
- Session-based behaviors
- Item IDs + categories
- Perfect for next-item prediction

------------------------------------------------------------------------

##  Preprocessing

-   Convert raw `.dat` files into indexed sequences
-   Pad sessions to fixed length (50)
-   Build next-item prediction pairs
-   Generate numpy-ready training arrays
-   Save mapping dictionaries (`item2idx`, etc.)

------------------------------------------------------------------------

##  EDA

Includes:
- Session length distribution
- Item popularity
- Category frequencies
- Long-tail analysis
- Interaction patterns

------------------------------------------------------------------------

##  Baseline Models: GRU4Rec

Three baselines trained:

  Model        Embedding   Hidden
  ------------ ----------- --------
  GRU_small    128         256
  GRU_medium   256         256
  GRU_large    256         512

------------------------------------------------------------------------

##  SASRec Transformer

Includes:

-   Masked self-attention
-   Positional encodings
-   Multi-head attention
-   Feed-forward transformer blocks
-   Tied embeddings
-   Dropout & LayerNorm

Engineering improvements: - Masking fixes → stable loss
- Gradient clipping
- LR scheduler
- Correct padding/causal attention

------------------------------------------------------------------------

##  Results (Example)

  Model        Recall@20   MRR@20
  ------------ ----------- -----------
  GRU (best)   ~0.11      ~0.029
  **SASRec**   **0.67+**   **0.37+**

------------------------------------------------------------------------

##  Trend Plots

-   Train vs Val Loss
-   Recall@10/20
-   MRR@10/20
##  How to Run

### 1. Preprocess:

``` bash
python src/preprocess_yoochoose.py
```

### 2. Train GRU:

``` bash
python src/train_gru.py
```

### 3. Train SASRec:

``` bash
python src/train_sasrec.py
```

### 4. Explore:

  Model Check.ipynb

------------------------------------------------------------------------

##  Future Work

-   Other Model improvement BERT4Rec Model , Next-basked Recommendation
-   FastAPI inference endpoint
-   MLflow tracking
-   Dockerized deployment
