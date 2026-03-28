# CNN-Based Hybrid Ring Artifact Reduction for CT Images
### Replication Study — Chen et al., IEEE TRPMS 2020

A liberal Python reimplementation of the hybrid Wavelet-Fourier + CNN ring artifact reduction algorithm described in:

> **Y. Chen et al., "A CNN-Based Hybrid Ring Artifact Reduction Algorithm for CT Images,"**  
> *IEEE Transactions on Radiation and Plasma Medical Sciences*, vol. 5, no. 2, pp. 198–206, Mar. 2021.  
> DOI: [10.1109/TRPMS.2020.2982401](https://ieeexplore.ieee.org/document/9047977)

The supplementary Wavelet-Fourier filtering algorithm is from:

> **B. Münch et al., "Stripe and ring artifact removal with combined wavelet-Fourier filtering,"**  
> *Optics Express*, vol. 17, no. 10, pp. 8567–8591, 2009.  
> [https://opg.optica.org/oe/fulltext.cfm?uri=oe-17-10-8567](https://opg.optica.org/oe/fulltext.cfm?uri=oe-17-10-8567)

---

## Overview

Ring artifacts in CT images are concentric bands caused by miscalibrated detector elements. This pipeline:

1. **Simulates ring artifacts** from artifact-free CT images (synthetic D_Norm via calibrated spikes)
2. **Applies WF precorrection** — Wavelet-Fourier sinogram-domain filtering before FBP reconstruction
3. **Trains a per-patient CNN** — 5-layer architecture that takes (x_Ring, x_PRE) as 2-channel input
4. **Produces the proposed method** — mutual-correlation patch fusion selecting the better pixel source
5. **Evaluates all stages** with ROI-based SSIM, quintuplet comparison figures, and CSV summary

---

## Files

| File | Run on | Description |
|------|--------|-------------|
| `01_create_ring_and_precorrected.ipynb` | **Colab** | Forward-projects artifact-free images, injects D_Norm ring artifacts, applies WF filter, saves `x_Ring` + `x_PRE` to Drive |
| `02_colab_train_cnn.ipynb` | **Colab** | Per-patient CNN training (25 epochs), inference on all slices, saves model checkpoints + `cnnimages/` to Drive (resume-safe after disconnects) |
| `03_create_proposed_method.py` | **Local** | Reads `x_Ring`, `x_PRE`, `x_CNN`; applies 9×9 mutual-correlation patch fusion (Eq. 8–11); writes `proposedmethod/` |
| `04_evaluate.py` | **Local** | ROI-SSIM evaluation with annular sampling, quintuplet comparison figures, per-ROI grouped bar charts, CSV summary |

---

## Data Requirements

You need the **AAPM 2016 Low-Dose CT Grand Challenge** dataset (full-dose, 1mm B30 kernel):

1. Download `1mm_B30/FD_1mm.zip` from:  
   **https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/145240708855**

2. Unzip and place on Google Drive with the following structure:
   ```
   MyDrive/
   └── medicalimaging/
       └── artifactfree/
           └── FD_1mm/
               └── full_1mm/
                   ├── L067/
                   │   └── full_1mm/
                   │       └── *.IMA
                   ├── L096/
                   │   └── full_1mm/
                   │       └── *.IMA
                   └── ...  (L109, L143, L192, L286, L291, L310, L333, L506)
   ```

---

## Pipeline Steps

### Step 1 — Generate ring artifacts + precorrected images (Colab)

Open `01_create_ring_and_precorrected.ipynb` in Google Colab.

- Reads from `MyDrive/medicalimaging/artifactfree/FD_1mm/`
- Writes to `MyDrive/medicalimaging/ringartifact/FD_1mm/` and `precorrected/FD_1mm/`
- Resume-safe: already-processed images are skipped on re-run

---

### Step 2 — Train CNNs + run inference (Colab)

Open `02_colab_train_cnn.ipynb` in Google Colab.

- **GPU runtime suggested**
- Reads ring + precorrected images from Drive
- Trains one dedicated model per patient (80/20 split within each patient)
- Saves to `MyDrive/Colab_Models/`:
  - `ring_cnn_{PID}.pt` — model checkpoint
  - `{PID}_training_curve.png` — log-scale MSE curve
  - `{PID}_ssim.png` — per-patient SSIM bar chart
- Saves CNN images to `MyDrive/medicalimaging/cnnimages/FD_1mm/`
- Resume-safe: a patient is skipped if all three output files already exist on Drive

> **Note on Colab disconnects:** The notebook checks `patient_is_done()` before processing each patient. If the session disconnects mid-patient, re-run the cell — that patient restarts from scratch, and previously completed patients are skipped.

---

### Step 3 — Download from Drive

Download the following folders from Google Drive to your local machine (preserving directory structure):

```
MyDrive/medicalimaging/ringartifact/
MyDrive/medicalimaging/precorrected/
MyDrive/medicalimaging/cnnimages/
```

Place them so your local directory looks like:
```
medicalimaging/
├── artifactfree/FD_1mm/...
├── ringartifact/FD_1mm/...
├── precorrected/FD_1mm/...
└── cnnimages/FD_1mm/...
```

---

### Step 4 — Generate proposed method images (Local)

```bash
pip install pydicom numpy scipy tqdm
python 03_create_proposed_method.py
```

Default paths: reads from `ringartifact/FD_1mm/`, `precorrected/FD_1mm/`, `cnnimages/FD_1mm/`; writes to `proposedmethod/FD_1mm/`.

Options:
```bash
python 03_create_proposed_method.py \
    --ring ringartifact/FD_1mm \
    --pre  precorrected/FD_1mm \
    --cnn  cnnimages/FD_1mm \
    --dst  proposedmethod/FD_1mm \
    --patch-size 9
```

---

### Step 5 — Evaluate (Local)

```bash
pip install pydicom numpy scikit-image matplotlib pandas tqdm
python 04_evaluate.py
```

Options:
```bash
python 04_evaluate.py \
    --ref       artifactfree/FD_1mm \
    --ring      ringartifact/FD_1mm \
    --pre       precorrected/FD_1mm \
    --cnn       cnnimages/FD_1mm \
    --proposed  proposedmethod/FD_1mm \
    --results   evaluation_results \
    --n-quintuplets 2 \
    --ssim-ymin 0.7 \
    --ssim-ymax 1.0
```

Outputs written to `evaluation_results/`:
```
evaluation_results/
├── aggregate_ssim.png
├── ssim_summary.csv
└── per_patient/
    ├── L067_ssim.png              # full-patient aggregate SSIM bar chart
    ├── L067_quintuplets.png       # 2-row × 5-col comparison figure with ROI overlay
    ├── L067_roi1_ssim.png         # grouped SSIM bar chart for ROI 1 (Fig. 9 style)
    ├── L067_roi2_ssim.png         # grouped SSIM bar chart for ROI 2
    ├── L067_quintuplet_ssim.png   # combined chart for all quintuplet ROIs
    └── ...                        # same for L096, L109, L143, L192, L286, L291, L310, L333, L506
```

---

## Software Requirements

```
Python >= 3.9
torch >= 2.0
scikit-image >= 0.21
PyWavelets
pydicom >= 2.4
scipy
numpy
matplotlib
pandas
tqdm
```

All Colab dependencies are installed automatically in the first cell of each notebook (`pip install -q ...`).

For local scripts:
```bash
pip install pydicom numpy scikit-image scipy matplotlib pandas tqdm
```
---

## Citation

If you use this code, please also cite the original paper:

```bibtex
@article{chen2020cnn,
  author  = {Chen, Yufei and Yin, Xindao and Shi, Lijuan and Shu, Huazhong and Luo, Limin and Coatrieux, Jean-Louis and Toumoulin, Christine},
  title   = {A {CNN}-Based Hybrid Ring Artifact Reduction Algorithm for {CT} Images},
  journal = {IEEE Transactions on Radiation and Plasma Medical Sciences},
  volume  = {5},
  number  = {2},
  pages   = {198--206},
  year    = {2021},
  doi     = {10.1109/TRPMS.2020.2982401}
}
```
