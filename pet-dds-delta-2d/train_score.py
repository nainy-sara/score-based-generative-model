"""
train_score.py
==============
Train the 2D score model on the Kaggle synthetic brain dataset.

This mirrors the paper's training setup:
  * an UNCONDITIONAL time-dependent score model s_theta(x, t),
  * trained with denoising score matching (Song & Ermon 2019),
  * on 2D transverse brain slices extracted from 3D full-count images.

The default hyperparameters match the paper's Section III where they
apply:  N = 100 diffusion steps.  Everything else (number of slices,
epochs, base channels) can be changed below without touching the model.

RUN ON KAGGLE
-------------
    !python train_score.py

OUTPUT
------
Best checkpoint is written to  {CHECKPOINT_DIR}/best*.ckpt
"""
import os
import shutil
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset

# Make sure local imports work regardless of the current working dir
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pet_dds import ScoreModelTrainer
from data_synthetic import load_synthetic_pet_slices, select_held_out_file
from paths import SYNTHETIC_BASE, CHECKPOINT_DIR, OUTPUT_DIR

# ------------------------- User settings -------------------------
NUM_SLICES         = 10000    # ~20 slices per volume for 500 volumes
BATCH_SIZE         = 32
MAX_EPOCHS         = 500      # EarlyStopping will stop earlier if converged
LEARNING_RATE      = 1e-4
EARLY_STOP_PATIENCE = 80
NUM_DIFFUSION_STEPS = 100     # paper Section III
BASE_CH            = 32
# -----------------------------------------------------------------


def main():
    pl.seed_everything(42)

    # Start from a clean checkpoint folder so the best*.ckpt we later load
    # genuinely belongs to this run.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.isdir(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---------- Dataset ----------
    held_out = select_held_out_file(SYNTHETIC_BASE)
    print(f"Held-out volume (never seen in training): {held_out}")

    slices = load_synthetic_pet_slices(
        SYNTHETIC_BASE,
        num_slices=NUM_SLICES,
        size=128,
        seed=0,
        exclude_files=[held_out],
    )
    print(f"Training tensor: {slices.shape}")

    loader = DataLoader(
        TensorDataset(slices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    # ---------- Model ----------
    model = ScoreModelTrainer(
        lr=LEARNING_RATE,
        N=NUM_DIFFUSION_STEPS,
        base_ch=BASE_CH,
    )

    # ---------- Callbacks ----------
    ckpt_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="best",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        save_last=True,
    )
    es_cb = EarlyStopping(
        monitor="train_loss",
        patience=EARLY_STOP_PATIENCE,
        mode="min",
    )

    # ---------- Training ----------
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accelerator,
        devices=1,
        callbacks=[ckpt_cb, es_cb],
        log_every_n_steps=20,
        enable_progress_bar=True,
    )

    print("Training...")
    trainer.fit(model, loader)
    print(f"Best checkpoint: {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()
