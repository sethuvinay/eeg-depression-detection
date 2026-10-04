# Multimodal Physiological Signal Analysis for Depression Research

A clean, well-documented reference implementation of neural-network methods for
detecting depressive physiological patterns from EEG and wearable biosignals —
combining **signal processing**, **graph neural networks (GNNs)** for cross-modal
fusion, and **self-supervised pretraining** on the public
[WESAD](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) dataset.

> **Note on provenance.** This repository is a clean-room *reference implementation*
> reconstructing the approach of a 2025 University at Albany research internship
> (*Machine Learning for Health and Emotion Recognition*). The original research
> code is no longer available, so the code here was written from the project's
> documented methods. The results quoted below are the **reported outcomes of the
> original research**, not of this reconstruction.

## Key results (original research)

| Result | Detail |
|---|---|
| **92% accuracy** | Classification accuracy on held-out multimodal data (subject-wise split) |
| **+31% cross-modal feature extraction** | Improvement from GNN fusion + self-supervised pretraining over the encoder baseline |

## Architecture

```
Raw WESAD signals ── chest @700 Hz (ECG, EDA, EMG, Resp) / wrist multi-rate
        │
        ▼
┌── Signal conditioning ──────────────────────────────┐
│ band-pass / notch filtering · resample → 64 Hz       │
│ per-subject z-normalisation · 60 s windows, 30 s stride│
└──────────────────────────────────────────────────────┘
        │
        ▼
┌── Per-modality 1D-CNN encoders ─────────────────────┐
│ ECG → 128-d │ EDA → 128-d │ EMG → 128-d │ Resp → 128-d │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌── Cross-modal GNN fusion ───────────────────────────┐
│ nodes = modalities · edges = learned affinities      │
│ 2-layer GCN message passing → fused representation   │
└──────────────────────────────────────────────────────┘
        │
        ▼
  Classifier → affective state (baseline / stress / amusement)
        ▲
  Self-supervised pretraining (contrastive, NT-Xent) on
  augmented signal views — before supervised fine-tuning
```

**Why this framing?** WESAD ships with experimentally induced *affective-state*
labels (baseline / stress / amusement), not clinical depression diagnoses. The
research used affective-state classification from physiological signals as a
benchmark proxy: dysregulated stress physiology is a well-studied correlate of
depressive disorders, so models that fuse multimodal biosignals robustly are
directly relevant to depression-screening applications.

## Dataset — WESAD

**WESAD** (Wearable Stress and Affect Detection; Schmidt et al., ICMI 2018):
15 subjects, chest-worn RespiBAN (ECG, EDA, EMG, respiration, temperature,
3-axis accel @ 700 Hz) + wrist-worn Empatica E4 (BVP @ 64 Hz, EDA/temp @ 4 Hz,
accel @ 32 Hz).

**Access:** the original release is distributed by the dataset providers and
**requires signing an End User License Agreement (EULA)** — request it via the
official dataset page: <https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html>.
A copy is also mirrored on the UCI Machine Learning Repository (dataset 465):
<https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection>.

Place the data so the layout is:

```
data/WESAD/
├── S2/S2.pkl
├── S3/S3.pkl
├── ...
└── S17/S17.pkl      # 15 subjects; S1 and S12 do not exist
```

The dataset itself is **not** committed to this repo (`data/` is gitignored).

Please cite the dataset if you use it:

> P. Schmidt, A. Reiss, R. Duerichen, C. Marberger, K. Van Laerhoven,
> "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect
> Detection", *Proc. ICMI*, Boulder, USA, 2018.

## Repository structure

```
eeg-depression-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                     # put WESAD here (gitignored, not committed)
├── notebooks/
│   └── demo_walkthrough.ipynb  # end-to-end demo of the pipeline
└── src/
    ├── __init__.py
    ├── data_loader.py        # WESAD .pkl loading, label maps, subject lists
    ├── preprocessing.py      # filtering, resampling, windowing, normalisation
    ├── features.py           # time/frequency-domain hand-crafted features
    ├── models.py             # CNN encoders, cross-modal GNN, SSL head, classifier
    ├── train.py              # SSL pretraining + supervised fine-tuning
    └── evaluate.py           # held-out evaluation, metrics, confusion matrix
```

## Quickstart

```bash
# 1. Create an environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) self-supervised pretraining of the modality encoders
python -m src.train --data-dir data/WESAD --phase ssl --epochs 50

# 3. Supervised fine-tuning with GNN fusion
python -m src.train --data-dir data/WESAD --phase supervised \
    --ssl-checkpoint checkpoints/ssl_best.pt --epochs 100

# 4. Evaluate on held-out subjects
python -m src.evaluate --data-dir data/WESAD \
    --checkpoint checkpoints/supervised_best.pt --plot outputs/confusion_matrix.png

# 5. Walk through the pipeline interactively
jupyter notebook notebooks/demo_walkthrough.ipynb
```

Useful flags for `train.py`: `--window-s`, `--stride-s`, `--target-fs`,
`--test-subjects` (comma-separated, e.g. `S16,S17`), `--batch-size`, `--lr`,
`--emb-dim`. Run `python -m src.train --help` for the full list.

## Methods summary

1. **Signal conditioning** (`preprocessing.py`) — zero-phase Butterworth
   band-pass/notch filtering per modality, polyphase resampling to a common
   64 Hz, per-subject z-normalisation, and 60 s sliding windows (30 s stride).
   Windows whose majority label is transient/undefined are discarded.
2. **Encoders** (`models.py`) — one 1D-CNN per modality (ECG, EDA, EMG, Resp)
   producing 128-d embeddings.
3. **Cross-modal GNN fusion** — modalities become graph nodes; a 2-layer GCN
   propagates information along *learned* inter-modality affinities, then mean
   pooling yields the fused representation.
4. **Self-supervised pretraining** — contrastive NT-Xent loss on two augmented
   views (jitter / scaling / temporal shift) of each window, warming up the
   encoders before any labels are used.
5. **Evaluation** — strictly subject-wise splits (no subject appears in both
   train and test), reporting accuracy, macro-F1, and a confusion matrix.

## Limitations & future work

- This is a reconstruction for portfolio purposes; hyperparameters and exact
  training curves of the original study are not reproduced here.
- WESAD affective states are a *proxy* task — validating on a clinically
  labelled depression dataset (e.g. DAIC-WOZ, with appropriate access) is the
  natural next step.
- The demo notebook's visualisations use synthetic signals where noted so they
  run without downloading the 2.5 GB dataset.
