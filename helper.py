"""Plotting, file-reading, and detection helpers.

Everything about "what does this capture show" lives here. Nothing in this
file talks to hardware -- see source.py for that. Fully standalone: does not
depend on any other project.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # save-to-file backend; works even with no display attached
import matplotlib.pyplot as plt  # noqa: E402

HALF_BANDWIDTH_HZ = 2.25e6       # Starlink/LTE 5MHz channel -> ~4.5MHz occupied
ELEVATION_THRESHOLD_DB = 3.0     # how far above background counts as "elevated"
BROAD_FRACTION_THRESHOLD = 0.5   # how much of the band must be elevated to call it "real data"


def load_iq(path):
    """Read a .cf32 file (raw interleaved float32 I/Q) plus its .json sidecar
    (written alongside it, holds sample_rate_hz and center_frequency_hz).

    Returns (samples, sample_rate_hz, center_freq_hz).
    """
    path = Path(path)
    samples = np.fromfile(path, dtype=np.complex64)
    meta_path = path.with_suffix(".json")
    fs, fc = 7.68e6, 0.0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        fs = meta.get("sample_rate_hz", fs)
        fc = meta.get("center_frequency_hz", fc)
    return samples, fs, fc


def save_iq(path, samples, sample_rate_hz, center_frequency_hz):
    """Write a .cf32 file plus its .json sidecar. `path` may be given with or
    without the .cf32 extension -- either way, the data lands at <path>.cf32."""
    path = Path(path).with_suffix(".cf32")
    np.asarray(samples, dtype=np.complex64).tofile(path)
    meta = {"sample_rate_hz": float(sample_rate_hz), "center_frequency_hz": float(center_frequency_hz),
            "n_samples": int(len(samples))}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def spectrum(samples, fs):
    """FFT power spectrum, 0 Hz placed at the centre. Returns (freqs_hz, power_db)."""
    spec = np.fft.fftshift(np.fft.fft(samples))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1.0 / fs))
    power_db = 20 * np.log10(np.abs(spec) + 1e-12)
    return freqs, power_db


def detect(path):
    """Look for a real, wide LTE/Starlink-shaped signal vs. a narrow spike vs. nothing.

    Returns a dict with the verdict and the numbers behind it (peak offset, levels)
    so you can inspect exactly what was found.
    """
    samples, fs, fc = load_iq(path)
    freqs, power_db = spectrum(samples, fs)

    in_band = np.abs(freqs) <= HALF_BANDWIDTH_HZ
    background_db = float(np.median(power_db[~in_band]))
    in_band_mean_db = float(np.mean(power_db[in_band]))
    elevated_fraction = float(np.mean(power_db[in_band] > background_db + ELEVATION_THRESHOLD_DB))

    height_ok = (in_band_mean_db - background_db) >= ELEVATION_THRESHOLD_DB
    breadth_ok = elevated_fraction >= BROAD_FRACTION_THRESHOLD

    peak_idx = int(np.argmax(power_db))
    peak_offset_hz = float(freqs[peak_idx])

    if height_ok and breadth_ok:
        verdict = "SIGNAL (wide, LTE-shaped)"
    elif height_ok:
        verdict = "SPIKE (narrow -- a tone or artifact, not spread-out data)"
    else:
        verdict = "NOTHING (just background noise)"

    return {
        "path": str(path),
        "center_freq_hz": fc,
        "verdict": verdict,
        "background_db": background_db,
        "in_band_mean_db": in_band_mean_db,
        "elevation_db": in_band_mean_db - background_db,
        "elevated_fraction": elevated_fraction,
        "peak_offset_hz": peak_offset_hz,
        "peak_freq_hz": fc + peak_offset_hz,
    }


def plot_pair(before_path, after_path, save_path, freq_label=""):
    """Side-by-side spectrum plot: 'before' (quiet) vs 'after' (should show a signal)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, path, title in ((ax1, before_path, "BEFORE (quiet)"), (ax2, after_path, "AFTER (transmitting)")):
        samples, fs, fc = load_iq(path)
        freqs, power_db = spectrum(samples, fs)
        ax.plot(freqs / 1e3, power_db, lw=0.7)
        ax.set_xlabel("offset from %.3f MHz (kHz)" % (fc / 1e6))
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    ax1.set_ylabel("magnitude (dB)")
    fig.suptitle("Signal check %s" % freq_label)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path
