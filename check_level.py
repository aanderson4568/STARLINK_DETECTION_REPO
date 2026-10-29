"""Check the HackRF calibration at a very low power, e.g. a -121 dBm generator tone.

calibrate_hackrf.py measures the offset with a fairly strong tone (-80 dBm). This script
proves it still holds down at Starlink levels: set the generator to a weak CW tone, run
this, and compare the measured dBm with the generator's setting. A pure tone can be seen
far below the noise by using very narrow frequency bins (a few Hz), averaged over time.

Uses the same frequency and gains as calibrate_hackrf.py, and its saved offset.
Run:  python check_level.py
"""

import json
from pathlib import Path

import numpy as np

import hackrf_rx
from calibrate_hackrf import (CABLE_LOSS_DB, GEN_FREQ_HZ, HACKRF_AMP, HACKRF_LNA, HACKRF_VGA,
                              RX_FREQ_HZ, SAMPLE_RATE_HZ)

GEN_LEVEL_DBM = -121.0   # what the generator is set to for this check

OUT_DIR = Path(__file__).parent / "out"


def main():
    cal = json.loads((OUT_DIR / "hackrf_cal.json").read_text())
    if (cal["rx_freq_hz"], cal.get("hackrf_amp"), cal.get("hackrf_lna"), cal.get("hackrf_vga")) != \
            (RX_FREQ_HZ, HACKRF_AMP, HACKRF_LNA, HACKRF_VGA):
        raise SystemExit("out/hackrf_cal.json was made with different settings -- run calibrate_hackrf.py first")
    fs = SAMPLE_RATE_HZ
    print("Recording 4 s with the generator at %.1f dBm, %.3f MHz..." % (GEN_LEVEL_DBM, GEN_FREQ_HZ / 1e6))
    path = hackrf_rx.capture(OUT_DIR / "check_tone", RX_FREQ_HZ, 4000.0, sample_rate_hz=fs,
                             lna=HACKRF_LNA, vga=HACKRF_VGA, amp=HACKRF_AMP)
    x = np.fromfile(path, dtype=np.complex64).astype(complex)
    x -= x.mean()

    n = 2 ** 21                                         # 3.7 Hz bins
    segs = x[:(len(x) // n) * n].reshape(-1, n)
    w = np.hanning(n)
    enbw = np.sum(w ** 2) / np.sum(w) ** 2 * n          # window's noise bandwidth, in bins
    spec = np.mean(np.abs(np.fft.fft(segs * w, axis=1)) ** 2, axis=0) / np.sum(w) ** 2
    f = np.fft.fftfreq(n, 1 / fs)
    k = int(np.argmin(np.abs(f - (GEN_FREQ_HZ - RX_FREQ_HZ))))
    k = k - 50 + int(np.argmax(spec[k - 50:k + 51]))    # the tone, allowing ~200 Hz of clock error
    noise_bin = np.median(spec[np.abs(f) > 2e3])
    snr = 10 * np.log10(spec[k] / noise_bin)
    if snr < 6:
        raise SystemExit("Tone not clearly visible (%.1f dB above noise) -- is the generator at %.3f MHz, CW, "
                         "RF on?" % (snr, GEN_FREQ_HZ / 1e6))
    measured = 10 * np.log10(spec[k - 3:k + 4].sum() / enbw) + cal["offset_db"]
    expected = GEN_LEVEL_DBM - CABLE_LOSS_DB
    noise_15k = 10 * np.log10(noise_bin * n / fs / enbw) + cal["offset_db"] + 10 * np.log10(15e3)
    print("  into the HackRF      %.1f dBm  (generator %.1f - cable %.1f)" % (expected, GEN_LEVEL_DBM, CABLE_LOSS_DB))
    print("  HackRF measured      %.1f dBm  (tone %.0f dB above noise in %.1f Hz bins)" % (measured, snr, fs / n))
    print("  error                %+.1f dB  %s" % (measured - expected, "GOOD" if abs(measured - expected) <= 1.5
                                                    else "-- larger than expected"))
    print("  noise per 15 kHz     %.1f dBm  -> an LTE subcarrier at %.0f dBm would be at SNR %.1f dB"
          % (noise_15k, expected, expected - noise_15k))


if __name__ == "__main__":
    main()
