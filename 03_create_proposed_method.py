"""
03_create_proposed_method.py
───────────────────────────────────────────────────────────────────────────────
Reads matched triples of images from:
    ringartifact/FD_1mm/      (x_Ring)
    precorrected/FD_1mm/      (x_PRE)
    cnnimages/FD_1mm/         (x_CNN)

Applies the mutual-correlation patch fusion described in
Chen et al. 2020, Section II-D to produce:
    proposedmethod/FD_1mm/    (x_C  — the proposed combined output)

Fusion algorithm
──────────────────────────────────────────
For each 9×9 pixel patch centred at (i, j):
  s_PRE(i,j) = x_Ring(i,j) − x_PRE(i,j)   (artifact component removed by WF)
  s_CNN(i,j) = x_Ring(i,j) − x_CNN(i,j)   (artifact component removed by CNN)
  Cor_PRE  = mutual_correlation(u_patch, s_PRE_patch)
  Cor_CNN  = mutual_correlation(u_patch, s_CNN_patch)
  if Cor_PRE ≥ Cor_CNN:
      x_C(i,j) = x_PRE(i,j)
  else:
      x_C(i,j) = x_CNN(i,j)

where u(i,j) = x_Ring(i,j) − x_free_est(i,j)  (ideal residual — approximated
by a mean of the two estimates when the artifact-free image is unavailable):
    u ≈ (s_PRE + s_CNN) / 2

Mutual correlation (Eq. 8):
    Cor(m, n) = 2 * <m, n> / (||m||^2 + ||n||^2  + ε)

Usage (run from the directory containing all set folders):
    python 04_create_proposed_method.py
    python 04_create_proposed_method.py --ring ringartifact/FD_1mm \\
                                        --pre  precorrected/FD_1mm \\
                                        --cnn  cnnimages/FD_1mm    \\
                                        --dst  proposedmethod/FD_1mm \\
                                        --patch-size 9

Dependencies: pydicom, numpy, scipy, tqdm
    pip install pydicom numpy scipy tqdm
"""

import argparse
from pathlib import Path

import numpy as np
import pydicom
from scipy.ndimage import uniform_filter
from tqdm import tqdm


# ── DICOM I/O ────────────────────────────────────────────────────────────────

def load_dicom_hu(path: Path):
    ds        = pydicom.dcmread(str(path))
    slope     = float(getattr(ds, 'RescaleSlope',     1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    hu        = ds.pixel_array.astype(np.float64) * slope + intercept
    return ds, hu


def save_dicom_hu(ds: pydicom.Dataset, new_hu: np.ndarray,
                  out_path: Path) -> None:
    slope     = float(getattr(ds, 'RescaleSlope',     1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    stored    = np.round((new_hu - intercept) / slope).astype(np.int16)

    new_ds = ds.copy()
    new_ds.PixelData           = stored.tobytes()
    new_ds.Rows                = stored.shape[0]
    new_ds.Columns             = stored.shape[1]
    new_ds.BitsAllocated       = 16
    new_ds.BitsStored          = 16
    new_ds.HighBit             = 15
    new_ds.PixelRepresentation = 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_ds.save_as(str(out_path))


# ── Mutual-correlation patch fusion ──────────────────────────────────────────

def mutual_correlation_maps(u: np.ndarray,
                             s_pre: np.ndarray,
                             s_cnn: np.ndarray,
                             patch_size: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-pixel mutual correlation maps using sliding-window
    box-filter approximation (fast, equivalent to the patch loop).

    Cor(m, n)[i,j] = 2 * E[m·n] / (E[m^2] + E[n^2] + ε)
    where E[·] is the mean over a patch_size × patch_size window.

    Returns (cor_pre, cor_cnn) — both shape (H, W).
    """
    eps = 1e-8
    f   = lambda a: uniform_filter(a, size=patch_size, mode='reflect')

    un_pre  = f(u  * s_pre)     # E[u · s_PRE]
    un_cnn  = f(u  * s_cnn)     # E[u · s_CNN]
    u2      = f(u  * u)         # E[u²]
    s_pre2  = f(s_pre * s_pre)  # E[s_PRE²]
    s_cnn2  = f(s_cnn * s_cnn)  # E[s_CNN²]

    cor_pre = (2.0 * un_pre) / (u2 + s_pre2 + eps)
    cor_cnn = (2.0 * un_cnn) / (u2 + s_cnn2 + eps)
    return cor_pre, cor_cnn


def fuse(x_ring: np.ndarray,
         x_pre:  np.ndarray,
         x_cnn:  np.ndarray,
         patch_size: int = 9) -> np.ndarray:
    """
    Produce the combined image x_C via mutual-correlation pixel selection

    When the artifact-free reference x_free is unavailable at inference
    time, the residual signal u (the true artifact component) is
    approximated as the average of the two method estimates:
        u ≈ (s_PRE + s_CNN) / 2
    This is the standard inference-time approximation.
    """
    # Removed artifact components (Eq. 9, 10)
    s_pre = x_ring - x_pre
    s_cnn = x_ring - x_cnn

    # Approximate u (Eq. 9 term — ideally x_Ring − x_free)
    u = (s_pre + s_cnn) / 2.0

    # Per-pixel mutual correlation (Eq. 8)
    cor_pre, cor_cnn = mutual_correlation_maps(u, s_pre, s_cnn, patch_size)

    # Pixel-level selection: pick the method with higher correlation (Eq. 11)
    x_combined = np.where(cor_pre >= cor_cnn, x_pre, x_cnn)
    return x_combined.astype(np.float64)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(ring_root: Path, pre_root: Path,
         cnn_root: Path,  dst_root: Path,
         patch_size: int) -> None:

    seen = set(); ring_files = []
    for pat in ('*.ima', '*.IMA', '*.dcm', '*.DCM'):
        for f in ring_root.rglob(pat):
            if f not in seen: seen.add(f); ring_files.append(f)
    ring_files = sorted(ring_files)
    if not ring_files:
        raise FileNotFoundError(f"No .ima/.IMA files found under {ring_root}")

    print(f"Found {len(ring_files)} ring-artifact images.")
    print(f"Patch size for fusion: {patch_size}×{patch_size}")
    print(f"Writing proposed-method images to: {dst_root}\n")

    missing_pre = 0
    missing_cnn = 0

    for ring_path in tqdm(ring_files, desc='Fusion', unit='img'):
        rel      = ring_path.relative_to(ring_root)
        pre_path = pre_root / rel
        cnn_path = cnn_root / rel
        out_path = dst_root / rel

        # Skip gracefully if corresponding files are missing.
        def find_file(p):
            if p.exists(): return p
            alt = p.with_suffix(p.suffix.swapcase())
            return alt if alt.exists() else None

        pre_found = find_file(pre_path)
        cnn_found = find_file(cnn_path)

        if pre_found is None:
            missing_pre += 1
            continue
        if cnn_found is None:
            missing_cnn += 1
            continue
        pre_path = pre_found
        cnn_path = cnn_found

        # --- Load all three images in HU ---
        ds_ring, hu_ring = load_dicom_hu(ring_path)
        _,       hu_pre  = load_dicom_hu(pre_path)
        _,       hu_cnn  = load_dicom_hu(cnn_path)

        # --- Fuse ---
        hu_combined = fuse(hu_ring, hu_pre, hu_cnn, patch_size=patch_size)

        # --- Save using ring-artifact DICOM metadata as template ---
        save_dicom_hu(ds_ring, hu_combined, out_path)

    n_done = len(ring_files) - missing_pre - missing_cnn
    print(f"\nDone. {n_done} proposed-method images written to {dst_root}")
    if missing_pre:
        print(f"  Warning: {missing_pre} precorrected files missing — those images skipped.")
    if missing_cnn:
        print(f"  Warning: {missing_cnn} CNN image files missing — those images skipped.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mutual-correlation fusion: proposed method images ')
    parser.add_argument('--ring', default='ringartifact/FD_1mm',
                        help='Ring Artifact Images root. Default: ringartifact/FD_1mm')
    parser.add_argument('--pre',  default='precorrected/FD_1mm',
                        help='Precorrected Images root. Default: precorrected/FD_1mm')
    parser.add_argument('--cnn',  default='cnnimages/FD_1mm',
                        help='CNN Images root. Default: cnnimages/FD_1mm')
    parser.add_argument('--dst',  default='proposedmethod/FD_1mm',
                        help='Output root. Default: proposedmethod/FD_1mm')
    parser.add_argument('--patch-size', type=int, default=9,
                        help='Patch size for mutual-correlation window. Default: 9')
    args = parser.parse_args()

    main(Path(args.ring).resolve(),
         Path(args.pre).resolve(),
         Path(args.cnn).resolve(),
         Path(args.dst).resolve(),
         patch_size=args.patch_size)
