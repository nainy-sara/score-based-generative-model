"""
paths.py
========
Central configuration of filesystem paths for the Kaggle run of the 2D
PET-DDS-delta reproduction of
    Webber et al. (2024), arXiv:2412.04319v1

WHY THIS FILE EXISTS
--------------------
On Kaggle the input dataset is mounted read-only under /kaggle/input, and
all writable outputs (checkpoints, figures) must go under /kaggle/working.
Keeping the two bases in one file makes the code portable between Kaggle
and a local machine: change these two strings and nothing else.
"""

# --- Read-only input: the synthetic brain FDG-PET dataset on Kaggle ---
# The NIfTI volumes live in the 'model_set' subfolder of the dataset.
# This matches the layout of the local copy at
#   F:\ai\data set\data set1\model_set
SYNTHETIC_BASE = (
    "/kaggle/input/datasets/lbvigilantdata/"
    "synthetic-neurodegenerative-brain-image-data-set/model_set"
)

# --- Writable output root: the notebook's working directory ---
OUTPUT_DIR     = "/kaggle/working/Andrew Reader/test2"
CHECKPOINT_DIR = OUTPUT_DIR + "/checkpoints"   # best*.ckpt is saved here
RESULTS_DIR    = OUTPUT_DIR + "/results"       # figures / PNGs are saved here
