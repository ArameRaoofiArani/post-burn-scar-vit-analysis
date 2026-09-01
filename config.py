"""Central configuration for the scar-quantification pipeline.

Every filesystem path used anywhere in this repository is defined here, and
every one of them can be overridden with an environment variable. Nothing
machine-specific is committed, so the scripts run unchanged on a different
computer once the variables below are set.

Quickest setup: put all the data under one folder and point SCAR_PROJECT_DIR
at it. The expected layout is then

    <SCAR_PROJECT_DIR>/
        predictions/                    <slide>_scar_head_arr.npy
                                        <slide>_scar_norm_arr.npy
        ground_truth/                   written by 03_extract_ground_truth.py
        scar_predictions/                    written by 02_classify_tiles.py
        slides/                         raw whole-slide images (.mrxs, .svs, ...)
        qupath_project/project.qpproj   QuPath project holding the annotations
        scar_centres_4.npy
        scar_centres_5.npy
        outputs/                        figures

Any individual folder can sit elsewhere -- set its own variable instead.

Two extra variables are needed only by the scripts that read raw slides
(03_extract_ground_truth.py, figure_tissue_panels.py):

    OPENSLIDE_BIN_DIR   Windows only: the "bin" folder of an OpenSlide
                        binary release, added to the DLL search path.
    PAQUO_QUPATH_DIR    the QuPath installation directory, read by paquo.

See README.md for a copy-pasteable example.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _path_from_env(var_name: str, default: Path) -> Path:
    """Path from an environment variable, falling back to a default."""
    value = os.environ.get(var_name)
    return Path(value).expanduser() if value else Path(default)


# --- which evaluation to run ----------------------------------------------
# The two configurations reported in the write-up. They are paired: each
# decision rule was scored over its own region definition, and mixing them
# gives numbers that match neither table.
#
#   baseline    argmax == scar, scored over a padded bounding box around
#               the annotation points.
#   threshold   percentile rank of the raw scar score above PCT_THRESHOLD,
#               scored over the connected tissue piece carrying the GT.
#
# Select with SCAR_EVALUATION=baseline or SCAR_EVALUATION=threshold.
# Scar predictions and metrics are written to a subfolder named for the choice,
# so running one never overwrites the other.
EVALUATIONS = {
    "baseline": {"rule": "argmax", "region": "bbox"},
    "threshold": {"rule": "percentile", "region": "component"},
}

EVALUATION = os.environ.get("SCAR_EVALUATION", "threshold").strip().lower()
if EVALUATION not in EVALUATIONS:
    raise ValueError(
        f"SCAR_EVALUATION={EVALUATION!r} is not recognised. "
        f"Choose one of: {', '.join(sorted(EVALUATIONS))}."
    )

RULE = EVALUATIONS[EVALUATION]["rule"]
REGION = EVALUATIONS[EVALUATION]["region"]

# The percentile cutoff the "threshold" arm operates at. Defined here rather
# than in 02_classify_tiles.py because the calibration and cluster-composition
# scripts need the identical value -- three separate copies had already drifted
# in precision.
#
# This is the TARGET-SENSITIVITY-0.70 operating point, not the Youden-optimal
# one. Both were evaluated on the pooled ROC across all six slides:
#
#   threshold 0.506   Youden-optimal (max sensitivity + specificity - 1),
#                     J = 0.483, but precision only 0.598.
#   threshold 0.561   target pooled sensitivity 0.700, specificity 0.772,
#                     precision 0.627, balanced accuracy 0.736, J = 0.472.
#
# 0.561 was adopted: it gives up very little J for a meaningfully better
# precision, and balances the three metrics most evenly. It is fitted on the
# same six slides it was evaluated on, so re-derive it for a new cohort --
# 05_calibrate_threshold.py is the script that does that.
PCT_THRESHOLD = 0.561174086


# --- directories -----------------------------------------------------------
PROJECT_DIR = _path_from_env("SCAR_PROJECT_DIR", REPO_ROOT / "data")

MODEL_OUTPUT_DIR = _path_from_env("SCAR_MODEL_OUTPUT_DIR", PROJECT_DIR / "predictions")
GT_DIR = _path_from_env("SCAR_GT_DIR", PROJECT_DIR / "ground_truth")
# One subfolder per evaluation -- the two configurations produce different
# grids for the same slide, so they must not share a filename.
SCAR_PREDICTION_DIR = _path_from_env(
    "SCAR_PREDICTION_DIR", PROJECT_DIR / "scar_predictions") / EVALUATION
SLIDE_DIR = _path_from_env("SCAR_SLIDE_DIR", PROJECT_DIR / "slides")
OUTPUT_DIR = _path_from_env("SCAR_OUTPUT_DIR", PROJECT_DIR / "outputs")

QUPATH_PROJECT = _path_from_env(
    "SCAR_QUPATH_PROJECT", PROJECT_DIR / "qupath_project" / "project.qpproj"
)
SCAR_CENTRES_4_PATH = _path_from_env(
    "SCAR_CENTRES_4", PROJECT_DIR / "scar_centres_4.npy"
)
SCAR_CENTRES_5_PATH = _path_from_env(
    "SCAR_CENTRES_5", PROJECT_DIR / "scar_centres_5.npy"
)


# --- dataset ---------------------------------------------------------------
# Slides to process. Override with SCAR_SLIDE_NAMES as a comma-separated list
# to run the pipeline on a different set without editing this file.
DEFAULT_SLIDE_NAMES = [
    "MSW02_09_08_d70",
    "MSW02_10_02_d70",
    "MSW02_10_04_d70",
    "MSW02_11_10_d70",
    "MSW02_11_12_d70",
    "MSW02_12_05_d70",
]
SLIDE_NAMES = [
    s.strip()
    for s in os.environ.get("SCAR_SLIDE_NAMES", ",".join(DEFAULT_SLIDE_NAMES)).split(",")
    if s.strip()
]


# --- model / pipeline constants -------------------------------------------
TILE_SIZE = 224

# Channel of the classification head holding the scar score. The four head
# channels are, in order: artefact, normal, scar, peri.
SCAR_CLASS_INDEX = 2
HEAD_CLASS_NAMES = {0: "artefact", 1: "normal", 2: "scar", 3: "peri"}

# Sub-clusters of scar_centres_5 retained as scar; the other two are the
# epidermis-like clusters discarded by 02_classify_tiles.py.
VALID_CLUSTER_INDICES = {2, 3, 4}

# Shared sub-cluster palette. Index 0 is "not scar"; index i + 1 is
# sub-cluster i, so the same colour means the same cluster in every figure.
CLUSTER_COLOURS = ["black", "khaki", "limegreen", "teal", "royalblue", "orangered"]


# --- optional native dependencies -----------------------------------------
def configure_openslide() -> None:
    """Make the OpenSlide DLLs importable on Windows.

    OpenSlide ships as a separate binary release; on Windows its "bin" folder
    must be on the DLL search path before `import openslide`. Call this
    before that import. On Linux and macOS the library comes from the system
    package manager, so this is a no-op.
    """
    if sys.platform != "win32":
        return
    dll_dir = os.environ.get("OPENSLIDE_BIN_DIR")
    if not dll_dir:
        raise RuntimeError(
            "OPENSLIDE_BIN_DIR is not set. On Windows, point it at the 'bin' "
            "folder of an OpenSlide binary release "
            "(https://openslide.org/download/) before running this script."
        )
    dll_path = Path(dll_dir).expanduser()
    if not dll_path.is_dir():
        raise FileNotFoundError(f"OPENSLIDE_BIN_DIR does not exist: {dll_path}")
    os.add_dll_directory(str(dll_path))


def configure_qupath() -> None:
    """Check that paquo can find a QuPath installation.

    paquo reads PAQUO_QUPATH_DIR itself, so this only fails early with a
    readable message instead of letting the import blow up later.
    """
    if not os.environ.get("PAQUO_QUPATH_DIR"):
        raise RuntimeError(
            "PAQUO_QUPATH_DIR is not set. Point it at your QuPath "
            "installation directory before running this script."
        )


def valid_mask_path(slide_name: str, region: str = None) -> Path:
    """Path to a slide's valid mask for the given region definition.

    03_extract_ground_truth.py writes both definitions, so switching
    SCAR_EVALUATION selects between existing files rather than needing the
    masks to be rebuilt.
    """
    return GT_DIR / f"{slide_name}_valid_mask_{region or REGION}.npy"


def gt_mask_path(slide_name: str) -> Path:
    """Path to a slide's ground-truth scar mask. The ground truth does not
    depend on the region definition -- only which tiles get scored does."""
    return GT_DIR / f"{slide_name}_gt_scar_mask.npy"


def ensure_dirs(*dirs: Path) -> None:
    """Create output directories if they don't exist yet."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
