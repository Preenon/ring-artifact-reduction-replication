"""
05_evaluate.py
───────────────────────────────────────────────────────────────────────────────
Evaluates all four processed image sets against the Artifact-Free reference
using ROI-based SSIM, produces one bar-chart per patient, and a final
aggregate chart averaged over all patients.

Additionally produces per-patient quintuplet figures: N slices selected near
the middle of each patient's scan are shown as a grid (N rows × 5 columns),
one column per image set, with the evaluated ROI overlaid as a yellow rectangle.

ROI definition
──────────────
For each image pair, a square ROI of area ≈ 1.5% of the full image is placed
at a reproducible random location within the top half of a circle of radius
R = min(H, W) // 2 centred on the image.  The ROI centre is seeded from the
relative file path (using hashlib.md5, not Python's hash(), which is
non-deterministic across runs) so the same position is used across all five
sets for any given image.

Outputs (all written to  evaluation_results/)
──────────────────────────────────────────────
  evaluation_results/
    per_patient/
      L067_ssim.png          ← SSIM bar chart
      L067_quintuplets.png   ← N-row × 5-col comparison figure
      ...
    aggregate_ssim.png
    ssim_summary.csv

Usage:
    python 05_evaluate.py
    python 05_evaluate.py --n-quintuplets 3   # 3 rows per patient figure
    python 05_evaluate.py --ref artifactfree/FD_1mm --results evaluation_results

Dependencies: pydicom, numpy, scikit-image, matplotlib, tqdm, pandas
    pip install pydicom numpy scikit-image matplotlib tqdm pandas
"""

import argparse
import csv
import math
import re
import hashlib
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)


# ── Constants ──────────────────────────────────────────────────────────────────

SET_KEYS = ['ring', 'pre', 'cnn', 'proposed']

SET_LABELS = {
    'ring':     'Ring artifact',
    'pre':      'Precorrected',
    'cnn':      'CNN image',
    'proposed': 'Proposed method',
}
# Column headers for quintuplet figure
SET_HEADERS = {
    'ref':      'Artifact-free image',
    'ring':     'Ring artifact image',
    'pre':      'Precorrected image',
    'cnn':      'CNN image',
    'proposed': 'Proposed method',
}
COLORS = {
    'ring':     '#e74c3c',
    'pre':      '#3498db',
    'cnn':      '#2ecc71',
    'proposed': '#9b59b6',
}

ROI_AREA_FRACTION  = 0.015   # 1.5% of image area
RING_INNER_FRAC    = 0.12    # inner radius of ring artifact zone  ≈ 61px on 512^2
RING_OUTER_FRAC    = 0.42    # outer radius of ring artifact zone  ≈ 215px on 512^2

# Standard soft-tissue display window (WL 40, WW 400)
# Displays HU directly — no μ conversion — giving full soft-tissue contrast.
# The paper uses [0, 0.5] cm ^ -1 which maps to [-1000, +1604] HU and washes out
# soft tissue because only ~15% of the scale covers the diagnostic range.
DISPLAY_VMIN = -160   # HU lower bound (WL 40 − WW/2)
DISPLAY_VMAX =  240   # HU upper bound (WL 40 + WW/2)


# ── DICOM loader ───────────────────────────────────────────────────────────────

def load_hu(path: Path) -> np.ndarray:
    ds        = pydicom.dcmread(str(path))
    slope     = float(getattr(ds, 'RescaleSlope',     1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    return ds.pixel_array.astype(np.float64) * slope + intercept


# ── ROI helpers ────────────────────────────────────────────────────────────────

def roi_size(H: int, W: int) -> int:
    """Side length (px) of the square ROI so its area ~= 1.5% of H×W."""
    return max(1, int(math.sqrt(ROI_AREA_FRACTION * H * W)))


def sample_roi_centre(H: int, W: int, seed: int) -> tuple[int, int]:
    """
    Return (row_c, col_c) sampled from the ring artifact annular zone.

    Ring artifacts are concentric circles centred on the image centre, spanning
    radii RING_INNER_FRAC–RING_OUTER_FRAC × min(H, W) (≈ 61–215 px on 512^2).
    Sampling uniformly in polar coordinates (radius, angle) over this annulus
    and all 360° gives high probability of landing directly on a ring artifact,
    regardless of which part of the body appears in the slice.

    The ROI is kept fully inside the image by clamping; rejection sampling
    ensures the clamped position is still within the annulus.
    """
    rng    = np.random.default_rng(seed)
    half_s = roi_size(H, W) // 2
    cx, cy = W // 2, H // 2
    r_inner = RING_INNER_FRAC * min(H, W)
    r_outer = RING_OUTER_FRAC * min(H, W)

    for _ in range(10_000):
        # Uniform sampling in annulus via sqrt trick (avoids centre bias)
        u     = rng.uniform(r_inner ** 2, r_outer ** 2)
        rad   = math.sqrt(u)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        row_c = int(round(cy + rad * math.sin(angle)))
        col_c = int(round(cx + rad * math.cos(angle)))

        # Keep ROI fully inside image bounds
        row_c = max(half_s, min(H - half_s - 1, row_c))
        col_c = max(half_s, min(W - half_s - 1, col_c))

        # Confirm the (possibly clamped) centre is still in the annulus
        dist = math.sqrt((row_c - cy) ** 2 + (col_c - cx) ** 2)
        if r_inner <= dist <= r_outer:
            return row_c, col_c

    # Fallback: place at 45° in the middle of the annulus
    mid_r = (r_inner + r_outer) / 2.0
    return (int(cy + mid_r * math.sin(math.pi / 4)),
            int(cx + mid_r * math.cos(math.pi / 4)))


def extract_roi(img: np.ndarray, r: int, c: int, side: int) -> np.ndarray:
    h = side // 2
    return img[r - h : r - h + side, c - h : c - h + side]


def path_seed(rel_path: Path) -> int:
    """Deterministic seed from file path (hashlib.md5, not hash() which is
    randomised per-process by PYTHONHASHSEED since Python 3.3)."""
    digest = hashlib.md5(str(rel_path).encode()).hexdigest()
    return int(digest, 16) % (2 ** 31)


# ── Patient ID ─────────────────────────────────────────────────────────────────

def patient_id_from_path(p: Path) -> str:
    for part in p.parts:
        if re.match(r'^L\d+$', part, re.IGNORECASE):
            return part.upper()
    return 'Unknown'


# ── Find a file with either extension case ─────────────────────────────────────

def find_path(root: Path, rel: Path) -> Path | None:
    p = root / rel
    if p.exists(): return p
    alt = p.with_suffix(p.suffix.swapcase())
    return alt if alt.exists() else None


# ── SSIM bar chart ─────────────────────────────────────────────────────────────

def bar_chart(means: dict, stds: dict, title: str, out_path: Path,
              y_min: float = 0.0, y_max: float = 1.0) -> None:
    keys   = SET_KEYS
    labels = [SET_LABELS[k] for k in keys]
    vals   = [means.get(k, 0.0) for k in keys]
    errs   = [stds.get(k,  0.0) for k in keys]
    cols   = [COLORS[k]         for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=cols, edgecolor='black',
                  linewidth=0.7, yerr=errs, capsize=4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('SSIM (ROI)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Quintuplet figure ──────────────────────────────────────────────────────────

def quintuplet_figure(ref_paths: list[Path],
                      set_roots: dict[str, Path],
                      ref_root:  Path,
                      pid:       str,
                      out_path:  Path) -> None:
    """
    Produce an N-row × 5-column comparison figure.

    Each row corresponds to one selected slice near the middle of the scan.
    Columns (left to right): artifact-free, ring artifact, precorrected, CNN, proposed method.

    Images are displayed in linear attenuation units (cm^-1), window [0, 0.5],
    matching the paper's stated display window.

    The evaluated ROI is overlaid on every panel as a yellow rectangle,
    labelled "ROI {row}" in the top-left corner (matching the paper's style).
    """
    n_rows = len(ref_paths)
    cols   = ['ref'] + SET_KEYS          # 5 columns
    n_cols = len(cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.2, n_rows * 3.2))
    # Normalise axes to always be 2-D array
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Column headers (only on the top row)
    for j, col_key in enumerate(cols):
        axes[0, j].set_title(SET_HEADERS[col_key], fontsize=9, pad=4)

    for row_idx, ref_path in enumerate(ref_paths):
        rel  = ref_path.relative_to(ref_root)
        seed = path_seed(rel)

        # Load reference image
        try:
            ref_hu = load_hu(ref_path)
        except Exception as e:
            print(f"  Quintuplet: could not load {ref_path.name}: {e}")
            for j in range(n_cols):
                axes[row_idx, j].axis('off')
            continue

        H, W = ref_hu.shape
        side = roi_size(H, W)
        max_attempts = 50
        for attempt in range(max_attempts):
            # change seed slightly each attempt to avoid identical sampling
            r, c = sample_roi_centre(H, W, seed + attempt)
            roi_ref = extract_roi(ref_hu, r, c, side)

            if not is_mostly_black(roi_ref, threshold=args.black_threshold):
                break
        else:
            # fallback: accept last ROI if all attempts fail
            pass
        half  = side // 2

        # Build the image list: [ref, ring, pre, cnn, proposed]
        images = [ref_hu]
        for col_key in SET_KEYS:
            p = find_path(set_roots[col_key], rel)
            try:
                images.append(load_hu(p) if p else None)
            except Exception:
                images.append(None)

        for j, (col_key, hu_img) in enumerate(zip(cols, images)):
            ax = axes[row_idx, j]

            if hu_img is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='grey')
                ax.axis('off')
                continue

            # Display in HU with standard soft-tissue window (WL 40, WW 400).
            # This preserves the full diagnostic contrast visible in a clinical
            # viewer — the [0, 0.5] cm^-1 μ window used by the paper maps to a
            # 2600 HU range that washes soft tissue into a uniform grey band.
            ax.imshow(hu_img, cmap='gray', vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX,
                      interpolation='none')
            ax.axis('off')

            # ── ROI rectangle overlay ──────────────────────────────────────
            # Rectangle: (x=col_left, y=row_top), width, height
            rect = mpatches.Rectangle(
                (c - half, r - half),       # (x, y) = (col, row) of top-left
                side, side,
                linewidth=1.5,
                edgecolor='yellow',
                facecolor='none',
            )
            ax.add_patch(rect)

            # ROI label in yellow at top-left of rectangle
            ax.text(c - half + 2, r - half + 2,
                    f'ROI {row_idx + 1}',
                    color='yellow', fontsize=7, fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(facecolor='none', edgecolor='none', pad=0))

            # Panel letter in bottom-right corner: (a), (b), …
            letter = chr(ord('a') + j)
            ax.text(0.97, 0.03, f'({letter})',
                    color='white', fontsize=8,
                    ha='right', va='bottom',
                    transform=ax.transAxes)

    fig.suptitle(f'Patient {pid}', fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Slice selection: N evenly-spaced indices near the centre ───────────────────

def select_middle_slices(sorted_paths: list[Path], n: int) -> list[Path]:
    """
    Select N paths evenly spaced within the middle 40% of sorted_paths
    (30th–70th percentile).  For N=1 this is exactly the median slice.

    Example: 560 images, N=2 → indices ~196 and ~364 (≈35% and 65%).
    """
    total = len(sorted_paths)
    if total == 0:
        return []
    if n >= total:
        return sorted_paths

    lo  = int(total * 0.30)
    hi  = int(total * 0.70)
    span = max(1, hi - lo)

    if n == 1:
        indices = [total // 2]
    else:
        indices = [round(lo + span * i / (n - 1)) for i in range(n)]

    # Clamp to valid range
    indices = [max(0, min(total - 1, idx)) for idx in indices]
    # Deduplicate while preserving order
    seen = set(); unique = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx); unique.append(idx)

    return [sorted_paths[i] for i in unique]


# ── SSIM computation for a specific list of slices ────────────────────────────

def compute_slice_ssims(ref_paths: list[Path],
                         set_roots: dict[str, Path],
                         ref_root:  Path) -> list[dict[str, float]]:
    """
    Compute ROI-based SSIM for each slice independently.

    Returns a list of dicts (one per slice), each mapping SET_KEYS → float.
    Missing counterpart files produce np.nan for that key/slice combination.
    """
    per_slice = []

    for ref_path in ref_paths:
        rel  = ref_path.relative_to(ref_root)
        seed = path_seed(rel)

        row: dict[str, float] = {k: np.nan for k in SET_KEYS}

        try:
            ref_hu = load_hu(ref_path)
        except Exception:
            per_slice.append(row)
            continue

        H, W = ref_hu.shape
        side = roi_size(H, W)
        max_attempts = 50
        for attempt in range(max_attempts):
            # change seed slightly each attempt to avoid identical sampling
            r, c = sample_roi_centre(H, W, seed + attempt)
            roi_ref = extract_roi(ref_hu, r, c, side)

            if not is_mostly_black(roi_ref, threshold=args.black_threshold):
                break
        else:
            # fallback: accept last ROI if all attempts fail
            pass
        data_range = ref_hu.max() - ref_hu.min()
        if data_range == 0:
            per_slice.append(row)
            continue

        for key, root in set_roots.items():
            p = find_path(root, rel)
            if p is None:
                continue
            try:
                other_hu = load_hu(p)
            except Exception:
                continue
            roi_other = extract_roi(other_hu, r, c, side)
            if roi_other.shape != roi_ref.shape:
                continue
            row[key] = float(ssim(roi_ref, roi_other, data_range=data_range))

        per_slice.append(row)

    return per_slice


# ── Grouped ROI bar chart (Fig. 9 style) ──────────────────────────────────────

def _grouped_roi_bar_chart(per_slice_list: list[dict[str, float]],
                            roi_labels:    list[str],
                            title:         str,
                            out_path:      Path,
                            y_min:         float = 0.0,
                            y_max:         float = 1.0) -> None:
    """
    Produce a grouped bar chart where each cluster on the x-axis represents
    one ROI (= one selected slice) and each bar within the cluster is one
    processing method.

    per_slice_list : list of {SET_KEY → float} dicts, one per ROI/slice.
    roi_labels     : matching list of x-axis cluster labels ('ROI 1', …).
    """
    n_rois    = len(per_slice_list)
    n_methods = len(SET_KEYS)
    bar_w     = 0.8 / n_methods          # total group width = 0.8
    x_centres = np.arange(n_rois, dtype=float)

    fig, ax = plt.subplots(figsize=(max(6, n_rois * 2.2), 5))

    for m_idx, key in enumerate(SET_KEYS):
        offsets = x_centres + (m_idx - (n_methods - 1) / 2.0) * bar_w
        vals    = [row.get(key, np.nan) for row in per_slice_list]
        # Replace nan with 0 for plotting only
        plot_vals = [v if not np.isnan(v) else 0.0 for v in vals]
        bars = ax.bar(offsets, plot_vals,
                      width=bar_w * 0.92,
                      color=COLORS[key],
                      edgecolor='black', linewidth=0.6,
                      label=SET_LABELS[key])
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (y_max - y_min) * 0.008,
                        f'{v:.4f}',
                        ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xticks(x_centres)
    ax.set_xticklabels(roi_labels, fontsize=11)
    ax.set_ylabel('SSIM', fontsize=11)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

def is_mostly_black(roi: np.ndarray,
                    threshold: float = 0.5,
                    hu_black: float = -900.0) -> bool:
    """
    Returns True if more than `threshold` fraction of pixels are "black".
    In CT HU, air ≈ -1000, so threshold like -900 works well.
    """
    frac_black = np.mean(roi < hu_black)
    return frac_black > threshold
    
# ── Main ───────────────────────────────────────────────────────────────────────

def main(ref_root:      Path,
         ring_root:     Path,
         pre_root:      Path,
         cnn_root:      Path,
         proposed_root: Path,
         results_dir:   Path,
         n_quintuplets: int,
         y_min:         float,
         y_max:         float) -> None:

    set_roots = {
        'ring':     ring_root,
        'pre':      pre_root,
        'cnn':      cnn_root,
        'proposed': proposed_root,
    }

    # ── Discover reference images (case-insensitive) ───────────────────────────
    seen_f = set(); ref_files = []
    for pat in ('*.ima', '*.IMA', '*.dcm', '*.DCM'):
        for f in ref_root.rglob(pat):
            if f not in seen_f: seen_f.add(f); ref_files.append(f)
    ref_files = sorted(ref_files)
    if not ref_files:
        raise FileNotFoundError(f"No .ima/.IMA files found under {ref_root}")
    print(f"Found {len(ref_files)} reference images.")

    # ── SSIM evaluation loop ───────────────────────────────────────────────────
    # Also track per-patient sorted file lists for quintuplet selection.
    patient_ssims: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    patient_files: dict[str, list[Path]] = defaultdict(list)

    for ref_path in tqdm(ref_files, desc='Evaluating', unit='img'):
        rel  = ref_path.relative_to(ref_root)
        pid  = patient_id_from_path(ref_path)
        seed = path_seed(rel)

        patient_files[pid].append(ref_path)

        try:
            ref_hu = load_hu(ref_path)
        except Exception as e:
            print(f"\nSkipping {ref_path.name}: {e}")
            continue

        H, W = ref_hu.shape
        side = roi_size(H, W)
        max_attempts = 50
        for attempt in range(max_attempts):
            # change seed slightly each attempt to avoid identical sampling
            r, c = sample_roi_centre(H, W, seed + attempt)
            roi_ref = extract_roi(ref_hu, r, c, side)

            if not is_mostly_black(roi_ref, threshold=args.black_threshold):
                break
        else:
            # fallback: accept last ROI if all attempts fail
            pass
        data_range = ref_hu.max() - ref_hu.min()

        for key, root in set_roots.items():
            p = find_path(root, rel)
            if p is None:
                continue
            try:
                other_hu = load_hu(p)
            except Exception:
                continue

            roi_other = extract_roi(other_hu, r, c, side)
            if roi_other.shape != roi_ref.shape:
                continue

            score = ssim(roi_ref, roi_other, data_range=data_range)
            patient_ssims[pid][key].append(score)

    if not patient_ssims:
        print("No SSIM scores computed — check that all set roots contain matching files.")
        return

    # ── Per-patient: SSIM bar chart + quintuplet figure ────────────────────────
    per_patient_dir = results_dir / 'per_patient'
    csv_rows = []
    all_patient_means: dict[str, list[float]] = defaultdict(list)

    for pid in sorted(patient_ssims.keys()):
        pid_ssims  = patient_ssims[pid]
        pid_paths  = sorted(patient_files[pid])   # chronological order

        means = {k: float(np.mean(v)) if v else 0.0
                 for k, v in pid_ssims.items()}
        stds  = {k: float(np.std(v))  if v else 0.0
                 for k, v in pid_ssims.items()}

        # SSIM bar chart (full patient, all slices)
        n_evaluated = min((len(v) for v in pid_ssims.values() if v), default=0)
        bar_chart(
            means, stds,
            title=f'Patient {pid} — Mean ROI SSIM per Method\n'
                  f'({n_evaluated} images evaluated)',
            out_path=per_patient_dir / f'{pid}_ssim.png',
            y_min=y_min, y_max=y_max)

        # Quintuplet figure + one bar chart per slice (Fig. 9 style)
        selected = select_middle_slices(pid_paths, n_quintuplets)
        if selected:
            quintuplet_figure(
                ref_paths = selected,
                set_roots = set_roots,
                ref_root  = ref_root,
                pid       = pid,
                out_path  = per_patient_dir / f'{pid}_quintuplets.png')

            # Per-slice raw SSIMs — one dict per slice, no averaging
            per_slice_scores = compute_slice_ssims(selected, set_roots, ref_root)

            for slice_idx, slice_scores in enumerate(per_slice_scores):
                roi_label   = f'ROI {slice_idx + 1}'
                slice_num   = pid_paths.index(selected[slice_idx]) + 1

                # Grouped bar chart: one cluster of 4 bars for this ROI
                # (With N quintuplets there are N clusters on the x-axis.)
                # For a single slice the chart has one cluster; the function
                # is designed to be called once per slice so each slice gets
                # its own saved file
                _grouped_roi_bar_chart(
                    per_slice_list  = [slice_scores],
                    roi_labels      = [roi_label],
                    title           = (f'Patient {pid} — {roi_label} SSIM\n'
                                       f'(slice {slice_num} of {len(pid_paths)})'),
                    out_path        = per_patient_dir
                                      / f'{pid}_roi{slice_idx + 1}_ssim.png',
                    y_min=y_min, y_max=y_max)

            # Also save one combined chart with ALL quintuplet ROIs together
            # (mirrors Fig. 9 directly when n_quintuplets > 1)
            if len(per_slice_scores) > 1:
                roi_labels_all = [f'ROI {i + 1}' for i in range(len(per_slice_scores))]
                _grouped_roi_bar_chart(
                    per_slice_list = per_slice_scores,
                    roi_labels     = roi_labels_all,
                    title          = (f'Patient {pid} — ROI SSIM '
                                      f'(quintuplet slices)'),
                    out_path       = per_patient_dir
                                     / f'{pid}_quintuplet_ssim.png',
                    y_min=y_min, y_max=y_max)

            print(f"  {pid}: SSIM done, quintuplet figure + "
                  f"{len(per_slice_scores)} ROI chart(s) saved.")
        else:
            print(f"  {pid}: SSIM done, no images for quintuplet figure.")

        row = {'Patient': pid}
        for k in SET_KEYS:
            row[SET_LABELS[k]]           = f'{means.get(k, 0):.6f}'
            row[SET_LABELS[k] + '_std']  = f'{stds.get(k, 0):.6f}'
            all_patient_means[k].append(means.get(k, 0.0))
        csv_rows.append(row)

    # ── Aggregate SSIM ─────────────────────────────────────────────────────────
    agg_means = {k: float(np.mean(v)) for k, v in all_patient_means.items() if v}
    agg_stds  = {k: float(np.std(v))  for k, v in all_patient_means.items() if v}

    bar_chart(
        agg_means, agg_stds,
        title=f'Aggregate — Second-Order Mean ROI SSIM\n'
              f'({len(patient_ssims)} patients)',
        out_path=results_dir / 'aggregate_ssim.png',
        y_min=y_min, y_max=y_max)

    # ── CSV ────────────────────────────────────────────────────────────────────
    agg_row = {'Patient': 'AGGREGATE'}
    for k in SET_KEYS:
        agg_row[SET_LABELS[k]]          = f'{agg_means.get(k,0):.6f}'
        agg_row[SET_LABELS[k] + '_std'] = f'{agg_stds.get(k,0):.6f}'
    csv_rows.append(agg_row)

    csv_path = results_dir / 'ssim_summary.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (['Patient'] +
                  [c for k in SET_KEYS
                   for c in (SET_LABELS[k], SET_LABELS[k] + '_std')])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nResults saved to  {results_dir}/")
    print(f"  {len(patient_ssims)} per-patient SSIM charts       → per_patient/*_ssim.png")
    print(f"  {len(patient_ssims)} quintuplet figures             → per_patient/*_quintuplets.png")
    print(f"  {len(patient_ssims)} quintuplet SSIM charts         → per_patient/*_quintuplet_ssim.png")
    print(f"  aggregate_ssim.png")
    print(f"  ssim_summary.csv")
    print(f"\nAggregate means:")
    for k in SET_KEYS:
        print(f"  {SET_LABELS[k]:20s}: {agg_means.get(k,0):.4f} ± {agg_stds.get(k,0):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ROI-SSIM evaluation + quintuplet comparison figures.')
    parser.add_argument('--ref',           default='artifactfree/FD_1mm')
    parser.add_argument('--ring',          default='ringartifact/FD_1mm')
    parser.add_argument('--pre',           default='precorrected/FD_1mm')
    parser.add_argument('--cnn',           default='cnnimages/FD_1mm')
    parser.add_argument('--proposed',      default='proposedmethod/FD_1mm')
    parser.add_argument('--results',       default='evaluation_results')
    parser.add_argument('--n-quintuplets', type=int, default=2,
                        help='Number of slices (rows) per patient quintuplet figure. '
                             'Slices are chosen evenly from the middle 40%% of the '
                             "patient's scan. Default: 2")
    parser.add_argument('--ssim-ymin', type=float, default=0.0,
                        help='Fixed lower bound of the SSIM y-axis on all bar charts. '
                             'Keeping this constant across all charts allows direct '
                             'visual comparison between patients. Default: 0.0')
    parser.add_argument('--ssim-ymax', type=float, default=1.0,
                        help='Fixed upper bound of the SSIM y-axis. Default: 1.0')
    parser.add_argument('--black-threshold', type=float, default=0.1,
                        help='Max allowed fraction of black pixels in ROI (default: 0.1)')
    args = parser.parse_args()

    main(ref_root      = Path(args.ref).resolve(),
         ring_root     = Path(args.ring).resolve(),
         pre_root      = Path(args.pre).resolve(),
         cnn_root      = Path(args.cnn).resolve(),
         proposed_root = Path(args.proposed).resolve(),
         results_dir   = Path(args.results).resolve(),
         n_quintuplets = args.n_quintuplets,
         y_min         = args.ssim_ymin,
         y_max         = args.ssim_ymax)
