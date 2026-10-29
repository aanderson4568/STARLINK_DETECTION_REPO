"""Pluto -> HackRF detection demo.

Fill in your settings below, then run:
    python main.py

What it does:
  1. Starts ONE continuous HackRF recording, CAPTURE_MS long.
  2. While it stays quiet, tunes+locks the Pluto onto FREQ_HZ but keeps it off-air --
     retuning causes a glitchy transient, so we get that out of the way now instead of
     at the moment we actually want the signal to appear.
  3. Once QUIET_MS has passed, starts streaming the tone -- since the Pluto is already
     locked onto the frequency, this is a clean, near-instant on-switch.
  4. Waits for the recording to finish, then stops the Pluto.
  5. Splits the recording into a first-half / second-half comparison, runs the
     detector on each half, and saves a side-by-side plot: before_after.png
  6. Opens an interactive GUI: drag a slider to scroll through the recording and
     watch the actual frequency-vs-amplitude spectrum change as the signal appears.
"""

import time
from pathlib import Path

import helper
import scroll_view
import source

# ---------------------------------------------------------------- SETTINGS: edit these
PLUTO_URI = "auto"               # "auto" finds the Pluto's USB address (it changes on replug);
                                  # or set one by hand, e.g. "usb:5.4.5" or "ip:192.168.2.3"
FREQ_HZ = 1900e6                 # frequency both radios use
PLUTO_ATTEN_DB = -50.0          # Pluto TX power. 0 = max power, more negative = less
HACKRF_GAIN_DB = 30.0            # HackRF RX gain. Raise if "nothing" even when transmitting;
                                  # lower if you get 100% clipping
CAPTURE_MS = 7000.0              # total length of the ONE continuous recording, in milliseconds
QUIET_MS = 3000.0                # how long it stays quiet before the tone switches on. Needs to
                                  # be long enough to cover the Pluto's ~2-2.5s retune/lock time,
                                  # which now happens during this window instead of at switch-on.
TONE_OFFSET_HZ = 100e3           # the test tone sits this far from FREQ_HZ
# ---------------------------------------------------------------------------------------

OUT_DIR = Path(__file__).parent / "out"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    pluto_uri = source.resolve_pluto_uri(PLUTO_URI)
    print("Pluto: %s" % pluto_uri)
    tone_path = source.make_tone_file(OUT_DIR / "tone", freq_offset_hz=TONE_OFFSET_HZ, center_freq_hz=FREQ_HZ)

    print("=" * 60)
    print("STEP 1: starting a %.1fs recording (quiet at first, tone switches on partway through)"
          % (CAPTURE_MS / 1000))
    t0 = time.time()
    capture = source.start_capture_hackrf(OUT_DIR / "capture", FREQ_HZ, HACKRF_GAIN_DB, CAPTURE_MS)

    print("STEP 2: locking the Pluto onto %.3f MHz (kept off-air) while staying quiet" % (FREQ_HZ / 1e6))
    source.configure_pluto_tx(pluto_uri, FREQ_HZ, atten_db=PLUTO_ATTEN_DB)
    # The saved recording's t=0 starts AFTER start_capture_hackrf()'s discarded lead-in
    # (see source.LEAD_IN_MS), so wait that much extra real time to land the on-switch
    # at QUIET_MS in the saved file, not QUIET_MS from this function call.
    target_wait_s = (QUIET_MS + capture["lead_in_ms"]) / 1000
    remaining_s = target_wait_s - (time.time() - t0)
    if remaining_s > 0:
        time.sleep(remaining_s)

    print("STEP 3: switching the tone on (clean on-switch -- already locked/tuned)")
    tx_proc = source.start_pluto_tx_stream(tone_path, pluto_uri)

    print("STEP 4: waiting for the recording to finish")
    capture_path = source.finish_capture_hackrf(capture)

    print("STEP 5: stopping Pluto")
    source.stop_pluto_tx(tx_proc)

    print("=" * 60)
    print("STEP 6: comparing the quiet portion to the transmitting portion")
    samples, fs, fc = helper.load_iq(capture_path)
    split = int(QUIET_MS / 1000 * fs)  # split exactly where the tone switched on, not just at 50%
    before_path = helper.save_iq(OUT_DIR / "before", samples[:split], fs, fc)
    after_path = helper.save_iq(OUT_DIR / "after", samples[split:], fs, fc)
    before = helper.detect(before_path)
    after = helper.detect(after_path)
    print("  quiet portion:        %s" % before["verdict"])
    print("  transmitting portion: %s" % after["verdict"])
    print("  peak (transmitting portion) at %.3f MHz (%.0f kHz from centre)" % (
        after["peak_freq_hz"] / 1e6, after["peak_offset_hz"] / 1e3))

    plot_path = helper.plot_pair(before_path, after_path, OUT_DIR / "before_after.png",
                                  freq_label="(%.1f MHz)" % (FREQ_HZ / 1e6))
    print("  saved: %s" % plot_path)

    print("=" * 60)
    print("SUMMARY")
    print("  quiet portion:        %s" % before["verdict"])
    print("  transmitting portion: %s" % after["verdict"])
    if before["verdict"].startswith("NOTHING") and after["verdict"].startswith(("SIGNAL", "SPIKE")):
        print("  -> Clean transition: off, then on. This is exactly the pattern you'd want to see")
        print("     for 'a satellite just came into view.'")

    print("=" * 60)
    print("STEP 7: opening the scrolling view of the FULL recording -- drag the slider past")
    print("        %.1fs to watch the signal switch on (close the window to end the program)"
          % (QUIET_MS / 1000))
    scroll_view.show(capture_path)


if __name__ == "__main__":
    main()
