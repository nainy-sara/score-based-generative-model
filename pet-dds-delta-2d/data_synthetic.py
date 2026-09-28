"""
data_synthetic.py
=================
NIfTI slice loader for the Kaggle synthetic brain dataset.

DATASET (from the paper's training regime, adapted)
---------------------------------------------------
The paper (Section III) trained the score model on 2D transverse brain
slices extracted from 3D full-count volumes of 55 subjects.  Here we use
the public Kaggle 'synthetic-neurodegenerative-brain-image-data-set'
which contains 500 volumes of the same shape -- one of them is held out
for reconstruction so the model never sees it during training.

The loader extracts random axial (z) slices, resizes them to 128x128,
augments with a horizontal flip and small intensity jitter, and finally
returns a (N, 1, 128, 128) float tensor in [0, 1].
"""
import os
import glob
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom


def find_nifti_files(base_dir):
    """Recursively find .nii / .nii.gz files, ignoring any *mask* files."""
    files = []
    for pat in ("**/*.nii", "**/*.nii.gz"):
        files.extend(glob.glob(os.path.join(base_dir, pat), recursive=True))
    files = [f for f in files if "mask" not in os.path.basename(f).lower()]
    return sorted(files)


def select_held_out_file(base_dir):
    """
    Deterministically pick ONE volume to reserve for the reconstruction
    test.  The same file must be passed to load_synthetic_pet_slices via
    the exclude_files argument so it is never seen during training.
    """
    files = find_nifti_files(base_dir)
    if not files:
        raise FileNotFoundError(f"No NIfTI files under {base_dir}")
    return files[-1]


def load_synthetic_pet_slices(base_dir, num_slices=10000, size=128, seed=0,
                              exclude_files=(), verbose=True):
    """
    Return a (num_slices, 1, size, size) tensor of random axial slices.

    Parameters
    ----------
    num_slices     total number of 2D slices to produce
    size           output side length (128 for this reproduction)
    seed           RNG seed for reproducibility
    exclude_files  list of absolute paths to skip (the held-out volume)
    """
    rng     = np.random.RandomState(seed)
    exclude = set(exclude_files)

    nii_files = [f for f in find_nifti_files(base_dir) if f not in exclude]
    if not nii_files:
        raise FileNotFoundError(
            f"No .nii files under {base_dir} after exclusion")

    if verbose:
        print(f"Found {len(nii_files)} training NIfTI volumes "
              f"(excluded {len(exclude)} held-out volume(s))")

    slices_per_volume = max(1, num_slices // len(nii_files))
    if verbose:
        print(f"Target: ~{slices_per_volume} slices per volume")

    all_slices = []
    for i, nii_path in enumerate(nii_files):
        try:
            img  = nib.load(nii_path)
            data = np.asarray(img.dataobj, dtype=np.float32)
            if data.ndim != 3:
                continue

            # Per-volume normalisation to [0, 1]
            vmin, vmax = data.min(), data.max()
            if vmax <= vmin:
                continue
            data = (data - vmin) / (vmax - vmin)

            num_axial = data.shape[2]
            for _ in range(slices_per_volume):
                z = rng.randint(0, num_axial)
                s = data[:, :, z].copy()

                # Resize / pad to (size, size)
                if s.shape != (size, size):
                    zf = (size / s.shape[0], size / s.shape[1])
                    s = zoom(s, zf, order=1)
                    s = s[:size, :size]
                    if s.shape != (size, size):
                        pad = np.zeros((size, size), dtype=np.float32)
                        pad[:s.shape[0], :s.shape[1]] = s
                        s = pad

                # Light augmentation
                if rng.rand() > 0.5:
                    s = s[:, ::-1].copy()
                s = s * rng.uniform(0.9, 1.1)
                s = np.clip(s, 0.0, 1.0)
                all_slices.append(s.astype(np.float32))

            if verbose and (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(nii_files)} volumes, "
                      f"{len(all_slices)} slices")

            if len(all_slices) >= num_slices:
                break

        except Exception as e:
            if verbose:
                print(f"  skip {os.path.basename(nii_path)}: {e}")
            continue

    if not all_slices:
        raise RuntimeError("No slices could be extracted.")

    arr = np.stack(all_slices[:num_slices], axis=0)
    if verbose:
        print(f"Final dataset: {arr.shape}, "
              f"range=[{arr.min():.3f}, {arr.max():.3f}]")
    return torch.tensor(arr).unsqueeze(1)
