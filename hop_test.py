"""Pluto -> HackRF hop test: quiet, then a tone that alternates +100 kHz / -100 kHz.

Run:
    python hop_test.py

What it does:
  1. Starts ONE continuous HackRF recording covering the whole schedule.
  2. Tunes+locks the Pluto onto FREQ_HZ (off-air) during the quiet stretch.
  3. Plays each tone in HOPS for HOP_MS, one after another. The Pluto stays tuned to
     FREQ_HZ the whole time -- only the baseband tone file changes -- so no retune glitch.
     Each swap restarts the Pluto stream, which leaves a short gap between hops.
  4. Measures the peak frequency over time, prints one line per hop, and saves
     out/hop_timeline.png (peak offset + power vs time).
  5. Opens the scrolling spectrum GUI over the full recording.
"""

import time
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import helper
import scroll_view
import source

# ---------------------------------------------------------------- SETTINGS: edit these
PLUTO_URI = "auto"               # "auto" finds the Pluto's USB address (it changes on replug);
                                  # or set one by hand, e.g. "usb:5.4.5" or "ip:192.168.2.3"
FREQ_HZ = 1995e6                 # !! 1990-1995 MHz is licensed (Starlink DTC / T-Mobile downlink).
                                  # Only use it with the Pluto CABLED straight to the HackRF, no
                                  # antennas. Over the air, use an ISM band (e.g. 915e6, 2.44e9).
PLUTO_ATTEN_DB = -50.0           # Pluto TX power. 0 = max power, more negative = less.
                                  # CABLED: never go above about -40 -- full power can damage the
                                  # HackRF's input. -50 + 30 dB gain: no clipping (checked).
HACKRF_GAIN_DB = 30.0            # HackRF RX (LNA) gain
HACKRF_RATE_HZ = 2e6             # +/-100 kHz fits easily; low rate keeps the file small
                                  # (a 25s recording at 7.68 Msps would be ~1.5 GB)
QUIET_MS = 3000.0                # silence before the first hop (covers the Pluto's lock time)
HOPS = [+100e3, -100e3, +100e3, -100e3]  # tone offsets from FREQ_HZ, played in order
HOP_MS = 5000.0                  # how long each tone stays on
TAIL_MS = 1000.0                 # quiet recording after the last hop
# ---------------------------------------------------------------------------------------

OUT_DIR = Path(__file__).parent / "out"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    pluto_uri = source.resolve_pluto_uri(PLUTO_URI)
    print("Pluto: %s" % pluto_uri)
    tone_paths = {off: source.make_tone_file(OUT_DIR / ("tone_%+dk" % (off / 1e3)), freq_offset_hz=off,
                                             center_freq_hz=FREQ_HZ)
                  for off in set(HOPS)}
    capture_ms = QUIET_MS + len(HOPS) * HOP_MS + TAIL_MS

    print("=" * 60)
    print("STEP 1: starting a %.1fs recording at %.3f MHz" % (capture_ms / 1000, FREQ_HZ / 1e6))
    t0 = time.time()
    capture = source.start_capture_hackrf(OUT_DIR / "hop_capture", FREQ_HZ, HACKRF_GAIN_DB, capture_ms,
                                          sample_rate_hz=HACKRF_RATE_HZ)
    lead_in_s = capture["lead_in_ms"] / 1000

    print("STEP 2: locking the Pluto (kept off-air) during the quiet stretch")
    source.configure_pluto_tx(pluto_uri, FREQ_HZ, atten_db=PLUTO_ATTEN_DB)

    print("STEP 3: hopping")
    tx_proc = None
    switch_times_s = []  # when each hop actually started, on the saved recording's clock
    for i, off in enumerate(HOPS):
        target_s = lead_in_s + (QUIET_MS + i * HOP_MS) / 1000
        remaining_s = target_s - (time.time() - t0)
        if remaining_s > 0:
            time.sleep(remaining_s)
        if tx_proc is not None:
            source.stop_pluto_tx(tx_proc, power_down=False)  # next tone takes over
        switch_times_s.append(time.time() - t0 - lead_in_s)
        print("  t=%5.2fs  tone %+.0f kHz" % (switch_times_s[-1], off / 1e3))
        tx_proc = source.start_pluto_tx_stream(tone_paths[off], pluto_uri)

    remaining_s = lead_in_s + (QUIET_MS + len(HOPS) * HOP_MS) / 1000 - (time.time() - t0)
    if remaining_s > 0:
        time.sleep(remaining_s)
    source.stop_pluto_tx(tx_proc)

    print("STEP 4: waiting for the recording to finish")
    capture_path = source.finish_capture_hackrf(capture)

    print("=" * 60)
    print("STEP 5: measuring the peak frequency over time")
    samples, fs, fc = helper.load_iq(capture_path)
    t_s, peak_hz, power_db = peak_track(samples, fs)
    noise_db = float(np.median(power_db[t_s < QUIET_MS / 1000 - 0.2]))

    print("  %-6s %-10s %-14s %-12s" % ("hop", "sent", "measured peak", "above quiet"))
    for i, off in enumerate(HOPS):
        start = QUIET_MS / 1000 + i * HOP_MS / 1000
        win = (t_s > start + 0.5) & (t_s < start + HOP_MS / 1000 - 0.2)  # skip the swap gap
        print("  %-6d %+6.0f kHz %+9.1f kHz %9.1f dB" % (
            i + 1, off / 1e3, np.median(peak_hz[win]) / 1e3, np.median(power_db[win]) - noise_db))

    png = plot_timeline(t_s, peak_hz, power_db, fc, OUT_DIR / "hop_timeline.png")
    print("  saved: %s" % png)

    print("=" * 60)
    print("STEP 6: opening the scrolling view -- drag through the recording to watch the tone")
    print("        jump between +%.0f and -%.0f kHz (close the window to end the program)"
          % (abs(HOPS[0]) / 1e3, abs(HOPS[0]) / 1e3))
    scroll_view.show(capture_path)


def peak_track(samples, fs, window_s=0.05, fft_size=4096):
    """Strongest frequency and its power in each `window_s` slice of the recording.

    Each slice averages several FFTs so the peak stands out from the noise.
    The DC bin (+/- a few bins) is ignored -- the HackRF has a spike there that
    isn't our signal.
    """
    n_win = int(window_s * fs)
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0 / fs))
    not_dc = np.abs(freqs) > 4 * fs / fft_size
    t_s, peak_hz, power_db = [], [], []
    for start in range(0, len(samples) - n_win + 1, n_win):
        chunk = samples[start:start + n_win][: (n_win // fft_size) * fft_size].reshape(-1, fft_size)
        psd = np.mean(np.abs(np.fft.fftshift(np.fft.fft(chunk, axis=1), axes=1)) ** 2, axis=0)
        k = int(np.argmax(np.where(not_dc, psd, 0)))
        t_s.append((start + n_win / 2) / fs)
        peak_hz.append(freqs[k])
        power_db.append(10 * np.log10(psd[k] + 1e-20))
    return np.array(t_s), np.array(peak_hz), np.array(power_db)


def plot_timeline(t_s, peak_hz, power_db, fc, save_path):
    fig = Figure(figsize=(12, 6))
    FigureCanvasAgg(fig)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.plot(t_s, peak_hz / 1e3, ".", ms=3)
    ax1.set_ylabel("peak offset (kHz)")
    ax1.set_title("Hop test at %.3f MHz" % (fc / 1e6))
    ax1.grid(True, alpha=0.3)
    ax2.plot(t_s, power_db, lw=0.9)
    ax2.set_ylabel("peak power (dB)")
    ax2.set_xlabel("time (s)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    return save_path


if __name__ == "__main__":
    main()
