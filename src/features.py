"""Hand-crafted time- and frequency-domain features.

These features serve two purposes in this project:

1. A classical baseline branch to sanity-check the learned representations.
2. Interpretable inputs for exploratory analysis (e.g. "which physiological
   markers separate stress from baseline?").

The deep pipeline in ``models.py`` learns its own features end-to-end; the
functions here are complementary, not competing.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy import signal as sp_signal


def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return a / (b + eps)


def time_domain_features(x: np.ndarray) -> Dict[str, float]:
    """Mean, spread, shape and Hjorth descriptors of a 1D window."""
    x = np.asarray(x, dtype=np.float64).ravel()
    dx = np.diff(x)
    var = np.var(x)
    var_dx = np.var(dx)

    activity = var
    mobility = np.sqrt(_safe_div(var_dx, var))
    ddx = np.diff(dx)
    complexity = _safe_div(np.sqrt(_safe_div(np.var(ddx), var_dx)), mobility)

    # Zero-crossing rate of the mean-centred signal.
    xc = x - x.mean()
    zcr = float(np.mean(xc[:-1] * xc[1:] < 0))

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "zcr": zcr,
        "hjorth_activity": float(activity),
        "hjorth_mobility": float(mobility),
        "hjorth_complexity": float(complexity),
    }


def frequency_domain_features(x: np.ndarray, fs: float) -> Dict[str, float]:
    """Welch PSD descriptors: band powers, spectral entropy, peak frequency.

    Bands follow the HRV-style convention (LF 0.04-0.15 Hz, HF 0.15-0.4 Hz),
    most meaningful for cardiac signals (ECG/BVP); for other modalities they
    still give a compact spectral summary.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    nperseg = min(256, len(x))
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg)
    total = psd.sum() + 1e-12

    def band_power(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(psd[mask].sum() / total)

    p = psd / total
    spectral_entropy = float(-np.sum(p * np.log(p + 1e-12)))
    peak_freq = float(freqs[np.argmax(psd)])

    lf = band_power(0.04, 0.15)
    hf = band_power(0.15, 0.40)

    return {
        "total_power": float(psd.sum()),
        "lf_power": lf,
        "hf_power": hf,
        "lf_hf_ratio": float(_safe_div(lf, hf)),
        "spectral_entropy": spectral_entropy,
        "peak_freq": peak_freq,
    }


def hrv_features(ecg: np.ndarray, fs: float) -> Dict[str, float]:
    """Simple heart-rate-variability descriptors from R-peak detection.

    Returns NaNs when too few peaks are found (short or noisy windows) — the
    caller decides how to handle them (impute or drop).
    """
    from scipy.signal import find_peaks

    ecg = np.asarray(ecg, dtype=np.float64).ravel()
    # Minimum plausible RR interval: 0.4 s (150 bpm).
    peaks, _ = find_peaks(ecg, distance=int(0.4 * fs),
                          prominence=0.3 * np.std(ecg))
    if len(peaks) < 3:
        return {"mean_rr": np.nan, "rmssd": np.nan, "sdnn": np.nan}

    rr = np.diff(peaks) / fs  # seconds
    diff_rr = np.diff(rr)
    return {
        "mean_rr": float(np.mean(rr)),
        "rmssd": float(np.sqrt(np.mean(diff_rr ** 2))),
        "sdnn": float(np.std(rr)),
    }


def extract_window_features(signals: Dict[str, np.ndarray],
                            fs: float) -> Tuple[np.ndarray, List[str]]:
    """Feature vector for one multi-modality window.

    ``signals`` maps modality name -> 1D conditioned window. ACC (3 axes) is
    reduced to its magnitude first. Returns ``(vector, feature_names)``.
    """
    feats: Dict[str, float] = {}
    for name, x in signals.items():
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:  # ACC -> magnitude
            x = np.linalg.norm(x, axis=-1)
        for k, v in time_domain_features(x).items():
            feats[f"{name}_td_{k}"] = v
        for k, v in frequency_domain_features(x, fs).items():
            feats[f"{name}_fd_{k}"] = v
        if name in ("ECG", "BVP"):
            for k, v in hrv_features(x, fs).items():
                feats[f"{name}_hrv_{k}"] = v
    names = sorted(feats)
    return np.array([feats[k] for k in names]), names
