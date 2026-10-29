"""Hardware control: starting/stopping the Pluto transmitter, and capturing with the HackRF.

This shells out to the same `iio_attr`/`iio_writedev`/`hackrf_transfer` native tools we
proved reliable throughout the original project, rather than re-implementing the low-level
SDR control from scratch.

Requires on PATH: iio_attr, iio_writedev, hackrf_transfer.
(On this machine those live under the miniforge install, e.g.
 C:\\Users\\avala\\miniforge3\\Library\\bin -- add that to PATH if these aren't found.)
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from helper import save_iq

# hackrf_transfer/iio_attr/iio_writedev live here on this machine. Add it to PATH so
# these are found no matter what launched this script (VS Code's Run button doesn't
# inherit the PATH additions we make in a manually-opened terminal).
_TOOLS_DIR = r"C:\Users\avala\miniforge3\Library\bin"
if _TOOLS_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _TOOLS_DIR + os.pathsep + os.environ.get("PATH", "")


def make_tone_file(path, freq_offset_hz=100e3, sample_rate_hz=7.68e6, duration_s=0.01, center_freq_hz=0.0):
    """Write a short, cleanly-looping tone .cf32 file to transmit from the Pluto."""
    n = int(sample_rate_hz * duration_s)
    t = np.arange(n) / sample_rate_hz
    x = (0.9 * np.exp(2j * np.pi * freq_offset_hz * t)).astype(np.complex64)
    return save_iq(path, x, sample_rate_hz, center_freq_hz)


def find_pluto_uri():
    """Return the Pluto's current USB address (e.g. "usb:5.4.5").

    The address changes whenever the Pluto is replugged, so look it up from
    `iio_info -s` instead of hard-coding it.
    """
    out = subprocess.run(["iio_info", "-s"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "PlutoSDR" in line and "[usb:" in line:
            return line[line.index("[usb:") + 1:line.index("]", line.index("[usb:"))]
    raise RuntimeError("No Pluto found over USB. `iio_info -s` printed:\n%s" % out)


def resolve_pluto_uri(uri):
    """"auto" -> look up the current USB address; anything else is used as given."""
    return find_pluto_uri() if uri == "auto" else uri


def configure_pluto_tx(uri, freq_hz, sample_rate_hz=7.68e6, atten_db=-20.0):
    """Tune the Pluto's transmitter to `freq_hz` and let its PLL lock/settle.

    Call this well *before* you actually want the tone on air (e.g. during the quiet
    part of a recording). Retuning causes a real glitchy transient while the AD9361's
    synthesizer re-locks -- doing that here, ahead of time, means start_pluto_tx_stream()
    (called later, at the moment you actually want the signal to appear) doesn't have to
    retune at all, so the on-switch itself is clean and near-instant instead of a ~1-2s mess.
    """
    timeout = 30000
    _iio_attr(uri, timeout, "-o", "-c", "ad9361-phy", "altvoltage1", "powerdown", "0")  # undo stop_pluto_tx
    _iio_attr(uri, timeout, "-o", "-c", "ad9361-phy", "altvoltage1", "frequency", str(int(freq_hz)))
    _iio_attr(uri, timeout, "-o", "-c", "ad9361-phy", "voltage0", "sampling_frequency", str(int(sample_rate_hz)))
    _iio_attr(uri, timeout, "-o", "-c", "ad9361-phy", "voltage0", "rf_bandwidth",
              str(int(min(sample_rate_hz, 20e6))))
    _iio_attr(uri, timeout, "-o", "-c", "ad9361-phy", "voltage0", "hardwaregain", str(float(atten_db)))
    time.sleep(1.5)  # let the PLL fully lock before we call this "ready"


def start_pluto_tx_stream(cf32_path, uri):
    """Start streaming `cf32_path` from an already-configured Pluto (see configure_pluto_tx).

    Returns a subprocess.Popen you must later pass to stop_pluto_tx().
    """
    samples = np.fromfile(cf32_path, dtype=np.complex64)

    timeout = 30000
    scaled = np.clip(np.round(samples * (2 ** 14)), -32768, 32767)
    inter = np.empty(2 * len(scaled), dtype="<i2")
    inter[0::2] = scaled.real.astype("<i2")
    inter[1::2] = scaled.imag.astype("<i2")
    raw_path = tempfile.NamedTemporaryFile(suffix=".raw", delete=False).name
    inter.tofile(raw_path)

    in_f = open(raw_path, "rb")
    cmd = ["iio_writedev", "-u", uri, "-T", str(timeout), "-c",
           "-b", str(len(scaled)), "-s", str(len(scaled)),
           "cf-ad9361-dds-core-lpc", "voltage0", "voltage1"]
    proc = subprocess.Popen(cmd, stdin=in_f, stderr=subprocess.PIPE)
    proc._raw_path = raw_path  # remember so stop_pluto_tx can clean it up
    proc._in_f = in_f
    proc._uri = uri

    # Verify it's actually still running before handing it back -- a bad push
    # (device busy, timed out, etc.) makes the process exit almost immediately,
    # and silently trusting a dead process here is exactly what caused a false
    # "before == after" result last time.
    time.sleep(2.0)
    if proc.poll() is not None:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        in_f.close()
        Path(raw_path).unlink(missing_ok=True)
        raise RuntimeError(
            "Pluto transmission failed to start (process exited immediately).\n"
            "stderr: %s\n"
            "Common cause: a leftover transmission from a previous run still holding the "
            "device -- check `tasklist | findstr iio_writedev` and kill any old ones, "
            "or unplug/replug the Pluto." % (err.strip() or "(none captured)"))
    return proc


def stop_pluto_tx(proc, power_down=True):
    """Stop a stream started with start_pluto_tx_stream().

    Killing iio_writedev alone does NOT silence the Pluto: its cyclic buffer keeps
    replaying the last tone in hardware. So by default this also powers down the TX
    synthesizer. Pass power_down=False when another stream starts right after (e.g.
    swapping tones mid-recording) -- the new stream replaces the buffer, and keeping
    the synthesizer locked avoids a retune glitch. configure_pluto_tx() powers it back up.
    """
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    proc._in_f.close()
    Path(proc._raw_path).unlink(missing_ok=True)
    if power_down:
        _iio_attr(proc._uri, 30000, "-o", "-c", "ad9361-phy", "altvoltage1", "powerdown", "1")


def capture_hackrf(out_cf32_path, freq_hz, gain_db=30.0, duration_ms=50.0, sample_rate_hz=7.68e6):
    """Record a snapshot with the HackRF, returns the path written."""
    n = int(duration_ms * 1e-3 * sample_rate_hz)
    tmp = tempfile.NamedTemporaryFile(suffix=".cs8", delete=False).name
    cmd = ["hackrf_transfer", "-r", tmp, "-f", str(int(freq_hz)), "-s", str(int(sample_rate_hz)),
           "-l", str(int(min(gain_db, 40))), "-g", "20", "-n", str(int(n * 1.2))]
    subprocess.run(cmd, check=True, capture_output=True)
    raw = np.fromfile(tmp, dtype=np.int8).astype(np.float32) / 128.0
    Path(tmp).unlink(missing_ok=True)
    x = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    x = x[int(0.1 * n):][:n]  # drop the startup transient
    return save_iq(out_cf32_path, x, sample_rate_hz, freq_hz)


# The first stretch of any hackrf_transfer capture is a USB startup transient -- bogus
# high-energy noise from the stream still stabilizing, not real RF. start_capture_hackrf
# records this much extra up front and finish_capture_hackrf() throws it away, so a
# capture that's supposed to start quiet actually starts quiet. How long this transient
# lasts varies run to run (seen anywhere from ~200ms to ~900ms) so this is set with a
# generous safety margin rather than the minimum observed.
LEAD_IN_MS = 1500.0


def start_capture_hackrf(out_cf32_path, freq_hz, gain_db=30.0, duration_ms=6000.0, sample_rate_hz=7.68e6,
                          lead_in_ms=LEAD_IN_MS):
    """Start a HackRF capture in the background and return immediately.

    Use this (instead of capture_hackrf) when something else needs to happen *during*
    the recording -- e.g. starting the Pluto partway through, so the single capture
    file itself contains the quiet-then-signal transition. Call finish_capture_hackrf()
    with the returned handle once you're ready to wait for it to complete.

    The saved file will be exactly `duration_ms` long, starting AFTER the discarded
    lead-in -- so if you're timing something else (like when the Pluto switches on)
    from the same moment you called this, add `lead_in_ms` to your own timer so it
    lines up with t=0 in the saved file, not with the real-world moment this was called.
    """
    lead_in_n = int(lead_in_ms * 1e-3 * sample_rate_hz)
    duration_n = int(duration_ms * 1e-3 * sample_rate_hz)
    tmp = tempfile.NamedTemporaryFile(suffix=".cs8", delete=False).name
    cmd = ["hackrf_transfer", "-r", tmp, "-f", str(int(freq_hz)), "-s", str(int(sample_rate_hz)),
           "-l", str(int(min(gain_db, 40))), "-g", "20", "-n", str(lead_in_n + duration_n)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"proc": proc, "tmp": tmp, "out_path": out_cf32_path, "freq_hz": freq_hz,
            "sample_rate_hz": sample_rate_hz, "lead_in_n": lead_in_n, "duration_n": duration_n,
            "lead_in_ms": lead_in_ms}


def finish_capture_hackrf(handle, timeout_s=60.0):
    """Wait for a capture started with start_capture_hackrf() to finish, decode it,
    trim off the startup transient, and save it as a .cf32 file. Returns the path written."""
    proc = handle["proc"]
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    raw = np.fromfile(handle["tmp"], dtype=np.int8).astype(np.float32) / 128.0
    Path(handle["tmp"]).unlink(missing_ok=True)
    x = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    x = x[handle["lead_in_n"]:][:handle["duration_n"]]  # drop the USB startup transient
    return save_iq(handle["out_path"], x, handle["sample_rate_hz"], handle["freq_hz"])


def _iio_attr(uri, timeout_ms, *args, attempts=4, delay_s=2.0):
    cmd = ["iio_attr", "-u", uri, "-T", str(timeout_ms)] + list(args)
    last_exc = None
    for attempt in range(attempts):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(delay_s)
    raise last_exc
