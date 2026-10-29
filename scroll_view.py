"""Interactive spectrum viewer: drag a slider to scroll through time and watch a single
frequency (x) vs amplitude (y) plot update live -- same style as before_after.png, but
scrollable instead of a fixed snapshot. It should sit flat near the noise floor before
the Pluto starts transmitting, then show a clear spike once it does.

Usage:
    python scroll_view.py out\\capture.cf32
    python scroll_view.py            (defaults to out\\after.cf32)
"""

import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
from scipy import signal

import matplotlib
matplotlib.use("TkAgg")  # the interactive, windowed backend (not the save-to-file one)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from helper import load_iq

# Frequency resolution of the spectrum -- fixed regardless of recording length, so a
# long recording doesn't blow up into a huge, slow FFT window.
FFT_SIZE = 1024

# Roughly how many time-slices to keep for the slider to scroll through. A short 50ms
# capture and a long several-second recording both end up with about this many, so the
# GUI stays responsive and the slider stays smooth either way.
TARGET_FRAMES = 1200


class ScrollingSpectrum:
    def __init__(self, root, path):
        self.samples, self.fs, self.fc = load_iq(path)
        self.total_ms = 1e3 * len(self.samples) / self.fs

        # Short-time FFT: one frequency spectrum per short time-slice, stacked up.
        nperseg = min(FFT_SIZE, len(self.samples))
        self.f, self.t, Sxx = signal.spectrogram(
            self.samples, fs=self.fs, nperseg=nperseg, noverlap=0, return_onesided=False)
        self.f = np.fft.fftshift(self.f)
        Sxx = np.fft.fftshift(Sxx, axes=0)

        # A long recording can produce far more time-slices than we need -- thin them
        # out evenly rather than holding/scrolling through all of them.
        if Sxx.shape[1] > TARGET_FRAMES:
            idx = np.linspace(0, Sxx.shape[1] - 1, TARGET_FRAMES).astype(int)
            Sxx = Sxx[:, idx]
            self.t = self.t[idx]

        self.power_db = 10 * np.log10(Sxx + 1e-12)  # shape: (n_freqs, n_time_frames)
        self.t_ms = self.t * 1e3
        self.f_khz = self.f / 1e3

        self.vmin = float(self.power_db.min())
        self.vmax = float(self.power_db.max())

        root.title("Scroll through time: %s" % path)
        root.geometry("1100x650")
        root.minsize(700, 450)

        self.fig = Figure(figsize=(12, 5.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        (self.line_plot,) = self.ax.plot(self.f_khz, self.power_db[:, 0], lw=0.9, color="tab:blue")
        self.ax.set_ylim(self.vmin - 2, self.vmax + 2)
        self.ax.set_xlim(self.f_khz.min(), self.f_khz.max())
        self.ax.set_xlabel("offset from %.3f MHz (kHz)" % (self.fc / 1e6))
        self.ax.set_ylabel("magnitude (dB)")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.slider = ttk.Scale(root, from_=0, to=self.total_ms, orient="horizontal", command=self._redraw)
        self.slider.pack(fill=tk.X, padx=10, pady=(10, 0))

        # Time labels under the slider, evenly spaced (0%, 25%, 50%, 75%, 100% of the
        # recording) so you can tell roughly where you are without reading the small text.
        ticks_frame = tk.Frame(root)
        ticks_frame.pack(fill=tk.X, padx=10)
        for i in range(5):
            ticks_frame.columnconfigure(i, weight=1)
            frac = i / 4.0
            tk.Label(ticks_frame, text="%.0f ms" % (frac * self.total_ms), font=("TkDefaultFont", 8),
                     anchor=("w" if i == 0 else "e" if i == 4 else "center")).grid(row=0, column=i, sticky="ew")

        self.label = tk.Label(root, text="", font=("TkDefaultFont", 10, "bold"))
        self.label.pack(pady=6)

        # Redo the layout whenever the window is resized, so the plot always matches
        # the widget's actual current size instead of the size it had when created.
        self.canvas.get_tk_widget().bind("<Configure>", self._on_resize)

        self._redraw(0)

    def _on_resize(self, event):
        self.canvas.draw_idle()

    def _redraw(self, t_ms):
        t_ms = float(t_ms)
        frame_idx = int(np.argmin(np.abs(self.t_ms - t_ms)))
        frame_t = self.t_ms[frame_idx]

        self.line_plot.set_ydata(self.power_db[:, frame_idx])
        self.canvas.draw_idle()

        self.label.config(text="t = %.2f ms  (%.1f / %.1f ms total) -- drag the slider to scroll" % (
            frame_t, frame_t, self.total_ms))


def show(path):
    """Open the scrolling spectrum GUI for `path`. Blocks until the window is closed."""
    root = tk.Tk()
    app = ScrollingSpectrum(root, path)  # noqa: F841 -- keep a reference alive
    root.mainloop()


if __name__ == "__main__":
    show(sys.argv[1] if len(sys.argv) > 1 else "out/after.cf32")
