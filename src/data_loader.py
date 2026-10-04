"""Loading utilities for the WESAD dataset.

WESAD (Wearable Stress and Affect Detection, Schmidt et al., ICMI 2018) ships
as one ``.pkl`` file per subject (``S2.pkl`` … ``S17.pkl``). Each file is a dict
with the following layout::

    {
        'subject': 'S2',
        'signal': {
            'chest': {'ECG': ..., 'EDA': ..., 'EMG': ..., 'Resp': ...,
                      'Temp': ..., 'ACC': ...},   # all @ 700 Hz
            'wrist': {'BVP': ..., 'EDA': ..., 'TEMP': ..., 'ACC': ...},
                      # BVP @ 64 Hz, ACC @ 32 Hz, EDA/TEMP @ 4 Hz
        },
        'label': np.ndarray,   # one label per chest sample (700 Hz)
    }

Labels: 0 = transient/undefined, 1 = baseline, 2 = stress, 3 = amusement.

Note on the task framing: WESAD provides experimentally induced *affective
state* labels rather than clinical depression diagnoses. This project treats
affective-state classification from physiological signals as a benchmark proxy
task — dysregulated stress physiology is a well-studied correlate of depressive
disorders — which is what makes the learned multimodal representations
relevant to depression-screening research.
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

#: Sampling rate of the chest-worn RespiBAN device (Hz).
CHEST_FS = 700

#: Sampling rates of the wrist-worn Empatica E4 signals (Hz).
WRIST_FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}

#: Chest signals available in every subject file.
CHEST_SIGNALS = ["ECG", "EDA", "EMG", "Resp", "Temp", "ACC"]

#: Wrist signals available in every subject file.
WRIST_SIGNALS = ["BVP", "EDA", "TEMP", "ACC"]

#: Raw WESAD label ids -> human-readable affective state.
LABEL_NAMES = {0: "transient", 1: "baseline", 2: "stress", 3: "amusement"}

#: Raw label ids usable for supervised training (transient windows are dropped).
USABLE_LABELS = (1, 2, 3)

#: Raw label id -> contiguous class index used by the classifier.
LABEL_TO_INDEX = {1: 0, 2: 1, 3: 2}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}

#: The 15 subject ids present in the public release (S1 and S12 do not exist).
VALID_SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]


def find_subject_files(data_dir: str) -> Dict[str, str]:
    """Map ``subject_id -> .pkl path`` for every subject found on disk.

    Accepts both ``<data_dir>/S2.pkl`` and ``<data_dir>/S2/S2.pkl`` layouts.
    """
    found: Dict[str, str] = {}
    for subject_id in VALID_SUBJECTS:
        candidates = [
            os.path.join(data_dir, f"{subject_id}.pkl"),
            os.path.join(data_dir, subject_id, f"{subject_id}.pkl"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                found[subject_id] = path
                break
    return found


def load_subject(data_dir: str, subject_id: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load one subject's chest signals and per-sample (700 Hz) labels.

    Returns
    -------
    signals : dict
        ``signal_name -> 1D/2D np.ndarray`` for the six chest modalities.
        ``ACC`` has shape ``(n_samples, 3)``; the rest are ``(n_samples,)``.
    labels : np.ndarray
        Integer label per chest sample (see :data:`LABEL_NAMES`).
    """
    files = find_subject_files(data_dir)
    if subject_id not in files:
        raise FileNotFoundError(
            f"No .pkl found for subject {subject_id} under {data_dir!r}. "
            "Download WESAD (requires a signed EULA, see README) and place it as "
            "data/WESAD/<subject>/<subject>.pkl."
        )
    with open(files[subject_id], "rb") as fh:
        raw = pickle.load(fh, encoding="latin1")

    chest = raw["signal"]["chest"]
    signals = {name: np.asarray(chest[name]) for name in CHEST_SIGNALS}
    labels = np.asarray(raw["label"]).astype(int).ravel()

    n = min(len(labels), min(s.shape[0] for s in signals.values()))
    signals = {k: v[:n] for k, v in signals.items()}
    return signals, labels[:n]


class WESADLoader:
    """Thin convenience wrapper around the module-level helpers."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.subject_files = find_subject_files(data_dir)
        if not self.subject_files:
            raise FileNotFoundError(
                f"No WESAD subject files found under {data_dir!r}."
            )

    @property
    def subjects(self) -> List[str]:
        """Subject ids available on disk, sorted."""
        return sorted(self.subject_files)

    def load(self, subject_id: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Load ``(signals, labels)`` for one subject."""
        return load_subject(self.data_dir, subject_id)
