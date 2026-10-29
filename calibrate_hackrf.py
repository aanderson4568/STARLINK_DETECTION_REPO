"""Calibrate the HackRF so its readings can be turned into real dBm.

Feed the HackRF a tone of KNOWN power from a signal generator, record it, and save
    offset_db = known power (dBm) - HackRF reading (dB full scale)
to out/hackrf_cal.json. decode.py then reports RSRP in dBm, like a phone does.

The offset is only valid for the SAME frequency, HackRF gain and sample rate as
lte_send.py uses -- change any of those and calibrate again.

Setup:
  1. Signal generator: CW (plain tone), GEN_FREQ_HZ, GEN_LEVEL_DBM, RF output OFF for now.
  2. Cable from the generator's RF OUT to the HackRF's ANTENNA port (no Pluto involved).
  3. Turn the generator's RF output ON, then run:  python calibrate_hackrf.py
  Safety: never put more than about -10 dBm into the HackRF (about -5 dBm max with the RF amp on).
"""

import json
import time
from pathlib import Path

import numpy as np

import hackrf_rx

# ---------------------------------------------------------------- SETTINGS: edit these
RX_FREQ_HZ = 1992.5e6        # must match FREQ_HZ in lte_send.py
HACKRF_AMP = True            # these three must match lte_send.py
HACKRF_LNA = 40
HACKRF_VGA = 30
SAMPLE_RATE_HZ = 7.68e6      # must match lte_send.py (its default)
GEN_FREQ_HZ = 1992.7e6       # generator tone: 200 kHz above centre, away from the HackRF DC spike
GEN_LEVEL_DBM = -80.0        # generator output level, as set on the generator
CABLE_LOSS_DB = 0.5          # loss of the cable/adapters at this frequency (~0.3-1 dB for a short SMA cable)
# ---------------------------------------------------------------------------------------

OUT_DIR = Path(__file__).parent / "out"


def tone_power(x, fs, f_tone, half_width_hz=20e3):
    """Power of the tone (fraction of full scale) and of the noise per Hz, from an FFT."""
    spec = np.abs(np.fft.fft(x)) ** 2 / len(x) ** 2           # Parseval: sum(spec) == mean(|x|^2)
    f = np.fft.fftfreq(len(x), 1 / fs)
    near = np.abs(f - f_tone) < half_width_hz
    noise_per_bin = np.median(spec[(np.abs(f) > 100e3) & ~near & (np.abs(f) < 3e6)])
    return spec[near].sum() - noise_per_bin * near.sum(), noise_per_bin * len(x) / fs


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Recording the generator tone (%.3f MHz at %.1f dBm)..." % (GEN_FREQ_HZ / 1e6, GEN_LEVEL_DBM))
    path = hackrf_rx.capture(OUT_DIR / "cal_tone", RX_FREQ_HZ, 1000.0, sample_rate_hz=SAMPLE_RATE_HZ,
                             lna=HACKRF_LNA, vga=HACKRF_VGA, amp=HACKRF_AMP)
    x = np.fromfile(path, dtype=np.complex64).astype(complex)
    x = x - x.mean()                                           # remove the DC offset

    biggest = max(np.abs(x.real).max(), np.abs(x.imag).max())
    p_tone, noise_per_hz = tone_power(x, SAMPLE_RATE_HZ, GEN_FREQ_HZ - RX_FREQ_HZ)
    if p_tone <= 0 or 10 * np.log10(p_tone / (noise_per_hz * 40e3)) < 20:
        raise SystemExit("No clear tone found -- is the generator RF output ON, at %.3f MHz, and cabled to "
                         "the HackRF?" % (GEN_FREQ_HZ / 1e6))
    if biggest > 0.9:
        raise SystemExit("HackRF is clipping (biggest sample %.2f) -- lower GEN_LEVEL_DBM and the generator "
                         "level by 10 dB and run again." % biggest)

    reading_db = 10 * np.log10(p_tone)
    into_hackrf_dbm = GEN_LEVEL_DBM - CABLE_LOSS_DB
    offset = into_hackrf_dbm - reading_db
    noise_dbm_hz = 10 * np.log10(noise_per_hz) + offset
    cal = {"offset_db": round(float(offset), 2), "rx_freq_hz": RX_FREQ_HZ, "hackrf_amp": HACKRF_AMP, "hackrf_lna": HACKRF_LNA, "hackrf_vga": HACKRF_VGA,
           "sample_rate_hz": SAMPLE_RATE_HZ, "gen_freq_hz": GEN_FREQ_HZ, "gen_level_dbm": GEN_LEVEL_DBM,
           "cable_loss_db": CABLE_LOSS_DB, "made": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT_DIR / "hackrf_cal.json").write_text(json.dumps(cal, indent=2))

    print("  power into the HackRF   %.1f dBm  (generator %.1f dBm - cable %.1f dB)"
          % (into_hackrf_dbm, GEN_LEVEL_DBM, CABLE_LOSS_DB))
    print("  HackRF reading          %.1f dB full scale   (biggest sample %.2f of 1.0)" % (reading_db, biggest))
    print("  OFFSET                  %+.2f dB   (real dBm = HackRF reading + offset)" % offset)
    print("  noise floor             %.1f dBm/Hz -> noise figure about %.0f dB (the -174 dBm/Hz ideal + NF)"
          % (noise_dbm_hz, noise_dbm_hz + 174))
    print("Saved out/hackrf_cal.json. Tip: run again at %.0f dBm (change both the generator and GEN_LEVEL_DBM);"
          % (GEN_LEVEL_DBM - 10))
    print("the offset should come out within about 1 dB if the HackRF is behaving linearly.")


if __name__ == "__main__":
    main()
