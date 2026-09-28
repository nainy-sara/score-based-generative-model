"""
paths.py
========
Central configuration of filesystem paths for the Kaggle run of the 2D
PET-DDS-delta reproduction of
    Webber et al. (2024), arXiv:2412.04319v1


SYNTHETIC_BASE = (
    "/kaggle/input/datasets/lbvigilantdata/"
    "synthetic-neurodegenerative-brain-image-data-set/model_set"
)

# --- Writable output root: the notebook's working directory ---
OUTPUT_DIR     = "/kaggle/...."
CHECKPOINT_DIR = OUTPUT_DIR + "/checkpoints"   # best*.ckpt is saved here
RESULTS_DIR    = OUTPUT_DIR + "/results"       # figures / PNGs are saved here
