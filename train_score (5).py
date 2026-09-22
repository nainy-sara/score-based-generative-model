"""train_score.py — trains the noise-prediction score model on real brain PET."""
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset

from pet_dds import ScoreModelTrainer
from data_utils import load_brain_pet_volume, make_real_brain_slices_2d


PET_DIR = "/kaggle/input/datasets/grantmcnatt/mri-and-pet-dice-similarity-dataset/data/BrainTumorPET"


def main():
    pl.seed_everything(42)
    print(f"Loading DICOM from {PET_DIR}...")
    volume = load_brain_pet_volume(PET_DIR, target_size=128)
    print(f"Volume: {volume.shape}")

    slices = make_real_brain_slices_2d(volume, num_slices=3000, size=128)
    print(f"Slices: {slices.shape}")

    loader = DataLoader(TensorDataset(slices), batch_size=32,
                        shuffle=True, num_workers=2)

    model = ScoreModelTrainer(lr=1e-4, N=100, base_ch=32)

    ckpt_cb = ModelCheckpoint(dirpath="checkpoints/", filename="best",
                              save_top_k=1, monitor="train_loss",
                              mode="min", save_last=True)
    es_cb = EarlyStopping(monitor="train_loss", patience=200, mode="min")

    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="auto",
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