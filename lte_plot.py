"""Plot the HackRF recording exactly as recorded -- nothing fixed or adjusted.

Top:    waterfall  -- time across, frequency up, colour = how strong
Bottom: spectrum   -- frequency across, strength up (averaged over the window shown)
Slider: drag to scroll through the recording. Only the window on screen is analysed,
        so scrolling stays fast.

Run:  python lte_plot.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import signal

RX_FILE = "out/lte_rx.cf32"
START_S = 1.5             # where the slider starts (s) -- LTE switches on at ~1 s
LENGTH_MS = 20.0          # how much to show at once. 20 ms = 2 LTE frames

# Load into memory and release the file right away, so an open plot window never stops
# lte_send.py from saving the next recording (Windows locks files that are kept open).
x = np.fromfile(RX_FILE, dtype=np.complex64)
meta = json.load(open(RX_FILE.replace(".cf32", ".json")))
fs, fc = meta["sample_rate_hz"], meta["center_frequency_hz"]
total_s = len(x) / fs
n_show = int(LENGTH_MS / 1000 * fs)

# Real power (dBm) needs the HackRF calibration from calibrate_hackrf.py, made at the same
# frequency and gains as this recording. Then: dBm = 10*log10(power as a fraction of full
# scale) + offset. Plotted per 15 kHz (one LTE subcarrier), the same bandwidth as RSRP.
SUBCARRIER_HZ = 15e3
cal_path = Path(RX_FILE).parent / "hackrf_cal.json"
cal = json.loads(cal_path.read_text()) if cal_path.exists() else None
gains = lambda d: (bool(d.get("hackrf_amp", False)), int(d.get("hackrf_lna", 24)) // 8 * 8,
                   int(d.get("hackrf_vga", 20)) // 2 * 2)
if cal and abs(cal["rx_freq_hz"] - fc) < 1e6 and gains(cal) == gains(meta):
    OFFSET_DB = cal["offset_db"] + 10 * np.log10(SUBCARRIER_HZ)
    UNIT = "dBm per 15 kHz"
    print("Calibrated: showing real power in dBm (calibration from %s)" % cal["made"])
else:
    OFFSET_DB = 0.0
    UNIT = "dB, relative -- not calibrated"
    print("No calibration for this frequency/gain -- showing relative dB. Run calibrate_hackrf.py "
          "with the same settings as lte_send.py to get dBm.")


def window(start_s):
    """Waterfall + spectrum for LENGTH_MS starting at start_s."""
    i = int(start_s * fs)
    chunk = np.asarray(x[i:i + n_show])
    # Short, overlapping pieces (128 samples = 0.017 ms, shorter than one 0.067 ms LTE symbol)
    f, t, power = signal.spectrogram(chunk, fs=fs, nperseg=128, noverlap=96, nfft=512,
                                     return_onesided=False)
    power_db = 10 * np.log10(np.fft.fftshift(power, axes=0) + 1e-30) + OFFSET_DB
    return np.fft.fftshift(f) / 1e6, (t + start_s) * 1000, power_db


# Fixed axis ranges, taken from windows spread across the whole recording,
# so quiet parts look quiet and loud parts look loud as you scroll.
samples = np.concatenate([window(s)[2].ravel() for s in np.linspace(0, total_s - LENGTH_MS / 1000, 8)])
vmin, vmax = np.percentile(samples, 20), np.percentile(samples, 99.5)

f, t, power_db = window(START_S)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
fig.subplots_adjust(bottom=0.14, hspace=0.35)

img = ax1.imshow(power_db, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                 extent=(t[0], t[-1], f[0], f[-1]))
ax1.set_xlabel("time (ms)")
ax1.set_ylabel("frequency (MHz from %.0f MHz)" % (fc / 1e6))
ax1.set_title("Waterfall: color = power (%s)" % UNIT)

# Average the power itself, then convert to dB (averaging dB values reads noise ~2.5 dB low).
mean_db = lambda p_db: 10 * np.log10(np.mean(10 ** (p_db / 10), axis=1))
(line,) = ax2.plot(f, mean_db(power_db))
ax2.set_ylim(vmin - 5, vmax + 5)
ax2.set_xlabel("frequency (MHz from %.0f MHz)" % (fc / 1e6))
ax2.set_ylabel("power (%s)" % UNIT)
ax2.set_title("Spectrum: average strength at each frequency (window shown above)")
ax2.grid(True)

slider = Slider(fig.add_axes([0.12, 0.03, 0.76, 0.03]), "start (s)",
                0, total_s - LENGTH_MS / 1000, valinit=START_S)


def update(start_s):
    f, t, power_db = window(start_s)
    img.set_data(power_db)
    img.set_extent((t[0], t[-1], f[0], f[-1]))
    line.set_ydata(mean_db(power_db))
    fig.canvas.draw_idle()


slider.on_changed(update)
plt.show()
