"""HackRF recording with all three gain stages adjustable.

Same as source.start_capture_hackrf / finish_capture_hackrf, plus the two gains that
source.py fixes (VGA = 20 dB, RF amp off):

  amp  RF amplifier at the input, off/on (about +11 dB). Biggest help for weak signals.
       With it on, keep the input below about -5 dBm or the HackRF can be damaged.
  lna  IF gain, 0-40 dB in steps of 8 (other values are rounded down).
  vga  baseband gain, 0-62 dB in steps of 2.

More gain lifts weak signals above the HackRF's own noise, but strong signals clip sooner.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

import source  # noqa: F401  (adds the hackrf_transfer folder to PATH)
from helper import save_iq

LEAD_IN_MS = 1500.0   # same USB startup transient handling as source.py


def start_capture(out_cf32_path, freq_hz, duration_ms, sample_rate_hz=7.68e6, lna=24, vga=20, amp=False):
    lead_in_n = int(LEAD_IN_MS * 1e-3 * sample_rate_hz)
    duration_n = int(duration_ms * 1e-3 * sample_rate_hz)
    tmp = tempfile.NamedTemporaryFile(suffix=".cs8", delete=False).name
    cmd = ["hackrf_transfer", "-r", tmp, "-f", str(int(freq_hz)), "-s", str(int(sample_rate_hz)),
           "-a", "1" if amp else "0", "-l", str(int(lna) // 8 * 8), "-g", str(int(vga) // 2 * 2),
           "-n", str(lead_in_n + duration_n)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"proc": proc, "tmp": tmp, "out_path": out_cf32_path, "freq_hz": freq_hz,
            "sample_rate_hz": sample_rate_hz, "lead_in_n": lead_in_n, "duration_n": duration_n,
            "lead_in_ms": LEAD_IN_MS}


def finish_capture(handle, timeout_s=60.0):
    proc = handle["proc"]
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    raw = np.fromfile(handle["tmp"], dtype=np.int8).astype(np.float32) / 128.0
    Path(handle["tmp"]).unlink(missing_ok=True)
    x = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    x = x[handle["lead_in_n"]:][:handle["duration_n"]]
    return save_iq(handle["out_path"], x, handle["sample_rate_hz"], handle["freq_hz"])


def capture(out_cf32_path, freq_hz, duration_ms, **gains):
    return finish_capture(start_capture(out_cf32_path, freq_hz, duration_ms, **gains))
