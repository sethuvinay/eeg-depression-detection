"""Signal conditioning for WESAD physiological signals.

Pipeline per chest modality: zero-phase Butterworth filtering (band-pass /
low-pass / notch as appropriate for the signal) -> polyphase resampling to a
common rate -> per-subject z-normalisation -> sliding windows.

All filters are applied with ``filtfilt`` (zero phase) so that temporal
alignment between modalities — which the cross-modal GNN relies on — is
preserved.
"""

from math import gcd
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal as sp_signal

#: Default per-modality filter recipes: (kind, low_hz, high_hz).
#: ``high_hz=None`` means low-pass; ``kind="notch"`` removes mains hum.
MODALITY_FILTERS = {
    "ECG": ("bandpass", 0.5, 40.0),    # keep QRS energy, drop drift & EMG bleed
    "EDA": ("lowpass", None, 5.0),     # electrodermal responses are very slow
    "EMG": ("bandpass", 20.0, 350.0),  # Nyquist is 350 Hz @ 700 Hz sampling
    "Resp": ("bandpass", 0.1, 0.8),    # normal breathing ~0.2-0.3 Hz
    "Temp": ("lowpass", None, 1.0),    # temperature drifts slowly
    "ACC": ("lowpass", None, 20.0),    # gross movement only
}


def butter_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 4):
    """Return (b, a) for a Butterworth band-pass filter."""
    nyq = fs / 2.0
    low, high = low_hz / nyq, high_hz / nyq
    return sp_signal.butter(order, [low, high], btype="band")


def butter_lowpass(cutoff_hz: float, fs: float, order: int = 4):
    """Return (b, a) for a Butterworth low-pass filter."""
    nyq = fs / 2.0
    return sp_signal.butter(order, cutoff_hz / nyq, btype="low")


def notch_coefficients(fs: float, freq: float = 50.0, q: float = 30.0):
    """Return (b, a) for a notch filter at mains frequency."""
    return sp_signal.iirnotch(freq, q, fs)


def apply_filter(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Zero-phase filtering along the last axis (preserves alignment)."""
    return sp_signal.filtfilt(b, a, x, axis=-1)


def filter_modality(x: np.ndarray, name: str, fs: float,
                    mains_freq: float = 50.0) -> np.ndarray:
    """Apply the recipe from :data:`MODALITY_FILTERS` plus a mains notch."""
    kind, low, high = MODALITY_FILTERS[name]
    if kind == "bandpass":
        b, a = butter_bandpass(low, high, fs)
    else:  # lowpass
        b, a = butter_lowpass(high, fs)
    x = apply_filter(x, b, a)
    b_n, a_n = notch_coefficients(fs, freq=mains_freq)
    return apply_filter(x, b_n, a_n)


def resample_signal(x: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Polyphase resampling to a common rate (anti-aliased)."""
    if orig_fs == target_fs:
        return x
    up = int(target_fs / gcd(int(orig_fs), int(target_fs)))
    down = int(orig_fs / gcd(int(orig_fs), int(target_fs)))
    return sp_signal.resample_poly(x, up, down, axis=-1)


def z_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel z-score normalisation (computed per subject)."""
    mu = x.mean(axis=-1, keepdims=True)
    sigma = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sigma + eps)


def sliding_windows(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Slice the last axis into overlapping windows.

    Returns an array of shape ``x.shape[:-1] + (n_windows, window)``.
    """
    n = x.shape[-1]
    if n < window:
        raise ValueError(f"signal length {n} shorter than window {window}")
    starts = np.arange(0, n - window + 1, stride)
    idx = starts[:, None] + np.arange(window)[None, :]
    return x[..., idx]


def preprocess_chest(signals: Dict[str, np.ndarray],
                     fs: float = 700,
                     target_fs: float = 64) -> Dict[str, np.ndarray]:
    """Filter, resample and normalise a subject's chest signals.

    Parameters
    ----------
    signals : dict
        Raw chest signals as returned by :func:`data_loader.load_subject`.
    fs : float
        Original sampling rate (700 Hz for RespiBAN).
    target_fs : float
        Common rate for the model (default 64 Hz keeps 60 s windows small
        enough for 1D-CNN encoders while preserving affective information).

    Returns
    -------
    dict
        ``signal_name -> (n_samples,)`` conditioned signals at ``target_fs``.
    """
    out: Dict[str, np.ndarray] = {}
    for name, x in signals.items():
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:  # ACC has shape (n, 3): filter each axis
            x = np.stack([filter_modality(x[:, c], name, fs) for c in range(x.shape[1])],
                         axis=1)
            x = np.stack([resample_signal(x[:, c], fs, target_fs) for c in range(x.shape[1])],
                         axis=1)
            x = z_normalize(x.T).T  # normalise per axis
        else:
            x = filter_modality(x, name, fs)
            x = resample_signal(x, fs, target_fs)
            x = z_normalize(x)
        out[name] = x
    return out


def window_labels(labels: np.ndarray, window: int, stride: int,
                  min_purity: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Assign one label per window by majority vote.

    Windows whose majority label is transient (0) or whose majority share is
    below ``min_purity`` are discarded. Returns ``(window_starts, labels)``.
    """
    wins = sliding_windows(labels.astype(int), window, stride)
    counts = np.apply_along_axis(lambda w: np.bincount(w, minlength=4), 1, wins)
    majority = counts.argmax(axis=1)
    purity = counts.max(axis=1) / window
    keep = (majority != 0) & (purity >= min_purity)
    starts = np.arange(0, len(labels) - window + 1, stride)[keep]
    return starts, majority[keep]


def build_window_index(n_samples_chest: int,
                       chest_fs: float,
                       target_fs: float,
                       window_s: float,
                       stride_s: float,
                       labels: np.ndarray) -> List[Tuple[int, int]]:
    """Map each kept window to ``(start_chest, start_resampled)`` indices.

    Labels live at the chest rate; conditioned signals at ``target_fs``.
    """
    window_chest = int(window_s * chest_fs)
    stride_chest = int(stride_s * chest_fs)
    starts_chest, _ = window_labels(labels, window_chest, stride_chest)
    ratio = target_fs / chest_fs
    return [(s, int(round(s * ratio))) for s in starts_chest]
