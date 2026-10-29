# Starlink Direct-to-Cell Detection

Tools for detecting and decoding Starlink Direct-to-Cell (D2C) LTE downlink signals
(1990–1995 MHz) with low-cost SDRs, as groundwork for RadioHound-based sensing.

A PlutoSDR stands in for the satellite and a HackRF One for the ground sensor, so the
whole chain (record, plot, detect, decode) can be tested before field captures.

## What's here

| File | What it does |
|---|---|
| `lte_send.py` | Pluto transmits an LTE waveform (from MATLAB); HackRF records it; opens the plot |
| `lte_plot.py` | Waterfall + spectrum with a time slider; real dBm when a matching calibration exists |
| `decode.py` | Single-script LTE decoder: PSS/SSS (cell ID, frequency offset), MIB, CFI, DCI (with RNTI search), PDSCH (turbo decoding, CRC check), plus RSRP and SNR |
| `calibrate_hackrf.py` | Calibrates the HackRF against a signal generator (power offset, noise figure) |
| `check_level.py` | Checks the calibration with a very weak tone (e.g. −121 dBm) |
| `hackrf_rx.py` | HackRF recording with adjustable RF amp, LNA and VGA gain |
| `matlab/make_lte_waveform.m` | Generates the LTE test waveform (RMC R.6) in MATLAB's LTE Toolbox |
| `source.py`, `helper.py` | Pluto/HackRF control and file/plot helpers |
| `main.py`, `hop_test.py`, `scroll_view.py` | Earlier tone detection and frequency-hop tests |
| `out/lte_matlab.cf32` | The MATLAB LTE waveform used in the tests (5 MHz, 25 RB, cell ID 378) |

Recordings are not included (hundreds of MB each).

## Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Also needed on PATH: `hackrf_transfer` (HackRF tools) and `iio_attr` / `iio_writedev` (libiio).

## Running

```bash
.venv\Scripts\python.exe lte_send.py      # send, record and plot
.venv\Scripts\python.exe decode.py        # decode the latest recording (out/lte_rx.cf32)
```

**1990–1995 MHz is licensed spectrum.** Transmit there only over a coax cable with no
antennas. For over-the-air tests set `FREQ_HZ = 915e6` (unlicensed 902–928 MHz).
Receiving real Starlink signals is receive-only.

## Results so far

At 1992.5 MHz over cable, the decoder recovered the cell ID and MIB down to −3 dB SNR,
and the scheduling messages down to 0 dB (Starlink D2C's typical SINR). At Starlink's
signal strength (≈ −121 dBm RSRP) the HackRF alone reaches only about −7.5 dB SNR, so an
external low-noise amplifier (LNA) is needed for field captures.
