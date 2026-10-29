"""Send the MATLAB LTE waveform with the Pluto and record it with the HackRF.

Run:  python lte_send.py      (opens the plots from lte_plot.py at the end)
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import hackrf_rx
import source

LTE_FILE = Path(__file__).parent / "out" / "lte_matlab.cf32"   # made in MATLAB
RX_FILE = Path(__file__).parent / "out" / "lte_rx"              # recording is saved here (.cf32)

PLUTO_URI = "auto"             # finds the USB address; or "ip:192.168.2.3" if only the network one shows up.
                               # "auto" finds the USB address instead.
FREQ_HZ = 1992.5e6        # Starlink D2C channel centre -- CABLE ONLY (licensed band). Antennas: use 915e6.
PLUTO_ATTEN_DB = -60.0    # Pluto TX power (0 = max, -89.75 = min). Stepping down toward Starlink level.
                          # With a cable go back to -30 (lower = less power).
HACKRF_AMP = True         # HackRF gains. Normal (strong signal, antennas ~1-2 ft): False / 24 / 20.
HACKRF_LNA = 40           # Weak signals (Starlink-level): True / 40 / 30 -- 17 dB better noise figure,
HACKRF_VGA = 30           # but a strong signal will clip. Amp on: keep input below about -5 dBm.
PLUTO_FIRST = True        # True: start the Pluto before recording (no quiet part, but avoids USB
                          # errors when both radios share a USB controller). False: quiet first.
QUIET_MS = 1000.0         # quiet at the start of the recording (only when PLUTO_FIRST = False)
RECORD_MS = 4000.0        # total recording length

pluto_uri = source.resolve_pluto_uri(PLUTO_URI)
print("Pluto:", pluto_uri)

if PLUTO_FIRST:
    # Set up the Pluto and start it looping BEFORE the HackRF records: while the HackRF
    # streams, commands to the Pluto over USB can fail (seen when both share a USB controller).
    # The recording is then LTE from start to finish, with no quiet part.
    print("Sending LTE...")
    source.configure_pluto_tx(pluto_uri, FREQ_HZ, atten_db=PLUTO_ATTEN_DB)
    tx = source.start_pluto_tx_stream(LTE_FILE, pluto_uri)
    print("Recording...")
    capture = hackrf_rx.start_capture(RX_FILE, FREQ_HZ, RECORD_MS, lna=HACKRF_LNA, vga=HACKRF_VGA, amp=HACKRF_AMP)
else:
    print("Recording...")
    t0 = time.time()
    capture = hackrf_rx.start_capture(RX_FILE, FREQ_HZ, RECORD_MS, lna=HACKRF_LNA, vga=HACKRF_VGA, amp=HACKRF_AMP)
    source.configure_pluto_tx(pluto_uri, FREQ_HZ, atten_db=PLUTO_ATTEN_DB)
    time.sleep(max(0, (QUIET_MS + capture["lead_in_ms"]) / 1000 - (time.time() - t0)))
    print("Sending LTE...")
    tx = source.start_pluto_tx_stream(LTE_FILE, pluto_uri)

try:
    rx_path = hackrf_rx.finish_capture(capture)
finally:
    source.stop_pluto_tx(tx)       # always switch the Pluto off, even if saving fails
# Note the settings in the recording's .json, so decode.py can match the calibration.
meta = json.loads(rx_path.with_suffix(".json").read_text())
meta.update(pluto_atten_db=PLUTO_ATTEN_DB, hackrf_amp=HACKRF_AMP, hackrf_lna=HACKRF_LNA, hackrf_vga=HACKRF_VGA)
rx_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
print("Done. Saved", rx_path)

# Plot it. lte_plot.py runs as its own program so its window can open
# (source.py loads helper.py, which switches matplotlib to save-to-file only).
print("Plotting...")
subprocess.run([sys.executable, "lte_plot.py"], cwd=Path(__file__).parent)
