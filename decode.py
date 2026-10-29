"""Decode an LTE downlink recording and print every setting that can be recovered from it.

For 5 MHz FDD LTE at 7.68 Msps, one antenna port (like MATLAB's R.6). It works the way a
phone does, one layer at a time:

  1. PSS        -> N_ID2, symbol timing, frequency offset
  2. SSS        -> N_ID1 (so the cell ID), frame start, FDD or TDD, normal or extended CP
  3. PBCH (MIB) -> bandwidth, PHICH settings, frame number, number of antenna ports
  4. PCFICH     -> CFI: how many OFDM symbols the control region uses
  5. PDCCH      -> each DCI (scheduling message) and the RNTI it is addressed to
  6. PDSCH      -> the transport block (user data), accepted only if its CRC passes

Adapted from lte-broadcast-sim's receiver (validated there against srsRAN and real
recordings) and extended for what R.6 uses: any RNTI, DCI format 1, 16/64QAM, and
transport blocks split into several turbo code blocks.

Run:  python decode.py                       decodes out/lte_rx.cf32 (the HackRF recording)
      python decode.py out/lte_matlab.cf32   decodes the MATLAB waveform directly
"""

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

FILE = "out/lte_rx.cf32"
START_S = 1.5             # where to start reading a long recording (LTE switches on at ~1 s)
N_FRAMES = 4              # how many 10 ms frames to analyse

# ------------------------------------------------------------------ 5 MHz LTE numbers
FS = 7.68e6
N_FFT, N_RB, N_SC = 512, 25, 300
CP = (40, 36, 36, 36, 36, 36, 36)                       # normal CP, symbols 0..6 of a slot
SYM_START = np.cumsum((0,) + tuple(c + N_FFT for c in CP[:-1]))  # CP start of each symbol
SLOT, SUBFRAME, FRAME = 3840, 7680, 76800
_k = np.arange(N_SC)
BINS = np.where(_k < N_SC // 2, _k - N_SC // 2, _k - N_SC // 2 + 1) % N_FFT  # DC skipped
CENTER72 = np.arange(N_SC // 2 - 36, N_SC // 2 + 36)  # PBCH / sync region
SYNC62 = np.arange(N_SC // 2 - 31, N_SC // 2 + 31)    # PSS / SSS subcarriers

# Turbo code block sizes and QPP interleaver parameters (36.212 Table 5.1.3-3) and the
# transport block size table for 1..25 PRB (36.213 Table 7.1.7.2.1-1, I_TBS 0..26),
# copied from lte-broadcast-sim/lte/tables.py (extracted there from srsRAN).
TURBO_K = [
    40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160,
    168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288,
    296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400, 408, 416,
    424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512, 528, 544, 560, 576,
    592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832,
    848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008, 1024, 1056, 1088, 1120, 1152,
    1184, 1216, 1248, 1280, 1312, 1344, 1376, 1408, 1440, 1472, 1504, 1536, 1568, 1600, 1632, 1664,
    1696, 1728, 1760, 1792, 1824, 1856, 1888, 1920, 1952, 1984, 2016, 2048, 2112, 2176, 2240, 2304,
    2368, 2432, 2496, 2560, 2624, 2688, 2752, 2816, 2880, 2944, 3008, 3072, 3136, 3200, 3264, 3328,
    3392, 3456, 3520, 3584, 3648, 3712, 3776, 3840, 3904, 3968, 4032, 4096, 4160, 4224, 4288, 4352,
    4416, 4480, 4544, 4608, 4672, 4736, 4800, 4864, 4928, 4992, 5056, 5120, 5184, 5248, 5312, 5376,
    5440, 5504, 5568, 5632, 5696, 5760, 5824, 5888, 5952, 6016, 6080, 6144,
]
TURBO_F1 = [
    3, 7, 19, 7, 7, 11, 5, 11, 7, 41, 103, 15, 9, 17, 9, 21,
    101, 21, 57, 23, 13, 27, 11, 27, 85, 29, 33, 15, 17, 33, 103, 19,
    19, 37, 19, 21, 21, 115, 193, 21, 133, 81, 45, 23, 243, 151, 155, 25,
    51, 47, 91, 29, 29, 247, 29, 89, 91, 157, 55, 31, 17, 35, 227, 65,
    19, 37, 41, 39, 185, 43, 21, 155, 79, 139, 23, 217, 25, 17, 127, 25,
    239, 17, 137, 215, 29, 15, 147, 29, 59, 65, 55, 31, 17, 171, 67, 35,
    19, 39, 19, 199, 21, 211, 21, 43, 149, 45, 49, 71, 13, 17, 25, 183,
    55, 127, 27, 29, 29, 57, 45, 31, 59, 185, 113, 31, 17, 171, 209, 253,
    367, 265, 181, 39, 27, 127, 143, 43, 29, 45, 157, 47, 13, 111, 443, 51,
    51, 451, 257, 57, 313, 271, 179, 331, 363, 375, 127, 31, 33, 43, 33, 477,
    35, 233, 357, 337, 37, 71, 71, 37, 39, 127, 39, 39, 31, 113, 41, 251,
    43, 21, 43, 45, 45, 161, 89, 323, 47, 23, 47, 263,
]
TURBO_F2 = [
    10, 12, 42, 16, 18, 20, 22, 24, 26, 84, 90, 32, 34, 108, 38, 120,
    84, 44, 46, 48, 50, 52, 36, 56, 58, 60, 62, 32, 198, 68, 210, 36,
    74, 76, 78, 120, 82, 84, 86, 44, 90, 46, 94, 48, 98, 40, 102, 52,
    106, 72, 110, 168, 114, 58, 118, 180, 122, 62, 84, 64, 66, 68, 420, 96,
    74, 76, 234, 80, 82, 252, 86, 44, 120, 92, 94, 48, 98, 80, 102, 52,
    106, 48, 110, 112, 114, 58, 118, 60, 122, 124, 84, 64, 66, 204, 140, 72,
    74, 76, 78, 240, 82, 252, 86, 88, 60, 92, 846, 48, 28, 80, 102, 104,
    954, 96, 110, 112, 114, 116, 354, 120, 610, 124, 420, 64, 66, 136, 420, 216,
    444, 456, 468, 80, 164, 504, 172, 88, 300, 92, 188, 96, 28, 240, 204, 104,
    212, 192, 220, 336, 228, 232, 236, 120, 244, 248, 168, 64, 130, 264, 134, 408,
    138, 280, 142, 480, 146, 444, 120, 152, 462, 234, 158, 80, 96, 902, 166, 336,
    170, 86, 174, 176, 178, 120, 182, 184, 186, 94, 190, 480,
]
TBS_TABLE = [
    [16, 32, 56, 88, 120, 152, 176, 208, 224, 256, 288, 328, 344, 376, 392, 424, 456, 488, 504, 536, 568, 600, 616, 648, 680],
    [24, 56, 88, 144, 176, 208, 224, 256, 328, 344, 376, 424, 456, 488, 520, 568, 600, 632, 680, 712, 744, 776, 808, 872, 904],
    [32, 72, 144, 176, 208, 256, 296, 328, 376, 424, 472, 520, 568, 616, 648, 696, 744, 776, 840, 872, 936, 968, 1000, 1064, 1096],
    [40, 104, 176, 208, 256, 328, 392, 440, 504, 568, 616, 680, 744, 808, 872, 904, 968, 1032, 1096, 1160, 1224, 1256, 1320, 1384, 1416],
    [56, 120, 208, 256, 328, 408, 488, 552, 632, 696, 776, 840, 904, 1000, 1064, 1128, 1192, 1288, 1352, 1416, 1480, 1544, 1608, 1736, 1800],
    [72, 144, 224, 328, 424, 504, 600, 680, 776, 872, 968, 1032, 1128, 1224, 1320, 1384, 1480, 1544, 1672, 1736, 1864, 1928, 2024, 2088, 2216],
    [328, 176, 256, 392, 504, 600, 712, 808, 936, 1032, 1128, 1224, 1352, 1480, 1544, 1672, 1736, 1864, 1992, 2088, 2216, 2280, 2408, 2472, 2600],
    [104, 224, 328, 472, 584, 712, 840, 968, 1096, 1224, 1320, 1480, 1608, 1672, 1800, 1928, 2088, 2216, 2344, 2472, 2536, 2664, 2792, 2984, 3112],
    [120, 256, 392, 536, 680, 808, 968, 1096, 1256, 1384, 1544, 1672, 1800, 1928, 2088, 2216, 2344, 2536, 2664, 2792, 2984, 3112, 3240, 3368, 3496],
    [136, 296, 456, 616, 776, 936, 1096, 1256, 1416, 1544, 1736, 1864, 2024, 2216, 2344, 2536, 2664, 2856, 2984, 3112, 3368, 3496, 3624, 3752, 4008],
    [144, 328, 504, 680, 872, 1032, 1224, 1384, 1544, 1736, 1928, 2088, 2280, 2472, 2664, 2792, 2984, 3112, 3368, 3496, 3752, 3880, 4008, 4264, 4392],
    [176, 376, 584, 776, 1000, 1192, 1384, 1608, 1800, 2024, 2216, 2408, 2600, 2792, 2984, 3240, 3496, 3624, 3880, 4008, 4264, 4392, 4584, 4776, 4968],
    [208, 440, 680, 904, 1128, 1352, 1608, 1800, 2024, 2280, 2472, 2728, 2984, 3240, 3368, 3624, 3880, 4136, 4392, 4584, 4776, 4968, 5352, 5544, 5736],
    [224, 488, 744, 1000, 1256, 1544, 1800, 2024, 2280, 2536, 2856, 3112, 3368, 3624, 3880, 4136, 4392, 4584, 4968, 5160, 5352, 5736, 5992, 6200, 6456],
    [256, 552, 840, 1128, 1416, 1736, 1992, 2280, 2600, 2856, 3112, 3496, 3752, 4008, 4264, 4584, 4968, 5160, 5544, 5736, 5992, 6200, 6456, 6968, 7224],
    [280, 600, 904, 1224, 1544, 1800, 2152, 2472, 2728, 3112, 3368, 3624, 4008, 4264, 4584, 4968, 5160, 5544, 5736, 6200, 6456, 6712, 6968, 7224, 7736],
    [328, 632, 968, 1288, 1608, 1928, 2280, 2600, 2984, 3240, 3624, 3880, 4264, 4584, 4968, 5160, 5544, 5992, 6200, 6456, 6712, 7224, 7480, 7736, 7992],
    [336, 696, 1064, 1416, 1800, 2152, 2536, 2856, 3240, 3624, 4008, 4392, 4776, 5160, 5352, 5736, 6200, 6456, 6712, 7224, 7480, 7992, 8248, 8760, 9144],
    [376, 776, 1160, 1544, 1992, 2344, 2792, 3112, 3624, 4008, 4392, 4776, 5160, 5544, 5992, 6200, 6712, 7224, 7480, 7992, 8248, 8760, 9144, 9528, 9912],
    [408, 840, 1288, 1736, 2152, 2600, 2984, 3496, 3880, 4264, 4776, 5160, 5544, 5992, 6456, 6968, 7224, 7736, 8248, 8504, 9144, 9528, 9912, 10296, 10680],
    [440, 904, 1384, 1864, 2344, 2792, 3240, 3752, 4136, 4584, 5160, 5544, 5992, 6456, 6968, 7480, 7992, 8248, 8760, 9144, 9912, 10296, 10680, 11064, 11448],
    [488, 1000, 1480, 1992, 2472, 2984, 3496, 4008, 4584, 4968, 5544, 5992, 6456, 6968, 7480, 7992, 8504, 9144, 9528, 9912, 10680, 11064, 11448, 12216, 12576],
    [520, 1064, 1608, 2152, 2664, 3240, 3752, 4264, 4776, 5352, 5992, 6456, 6968, 7480, 7992, 8504, 9144, 9528, 10296, 10680, 11448, 11832, 12576, 12960, 13536],
    [552, 1128, 1736, 2280, 2856, 3496, 4008, 4584, 5160, 5736, 6200, 6968, 7480, 7992, 8504, 9144, 9912, 10296, 11064, 11448, 12216, 12576, 12960, 13536, 14112],
    [584, 1192, 1800, 2408, 2984, 3624, 4264, 4968, 5544, 5992, 6712, 7224, 7992, 8504, 9144, 9912, 10296, 11064, 11448, 12216, 12960, 13536, 14112, 14688, 15264],
    [616, 1256, 1864, 2536, 3112, 3752, 4392, 5160, 5736, 6200, 6968, 7480, 8248, 8760, 9528, 10296, 10680, 11448, 12216, 12576, 13536, 14112, 14688, 15264, 15840],
    [712, 1480, 2216, 2984, 3752, 4392, 5160, 5992, 6712, 7480, 8248, 8760, 9528, 10296, 11064, 11832, 12576, 13536, 14112, 14688, 15264, 16416, 16992, 17568, 18336],
]


# ================================================================== sequences (36.211)

@lru_cache(maxsize=None)
def gold(c_init, n):
    """Pseudo-random sequence c(0..n-1), 36.211 7.2."""
    x1 = np.zeros(n + 1600 + 31, np.uint8)
    x2 = np.zeros(n + 1600 + 31, np.uint8)
    x1[0] = 1
    x2[:31] = [(c_init >> i) & 1 for i in range(31)]
    for i in range(n + 1600):
        x1[i + 31] = x1[i + 3] ^ x1[i]
        x2[i + 31] = x2[i + 3] ^ x2[i + 2] ^ x2[i + 1] ^ x2[i]
    return x1[1600:1600 + n] ^ x2[1600:1600 + n]


def pss_seq(n_id2):
    u = (25, 29, 34)[n_id2]
    n = np.arange(62)
    return np.where(n <= 30, np.exp(-1j * np.pi * u * n * (n + 1) / 63),
                    np.exp(-1j * np.pi * u * (n + 1) * (n + 2) / 63))


def _mseq(taps):
    x = [0, 0, 0, 0, 1]
    for i in range(26):
        x.append(sum(x[i + t] for t in taps) % 2)
    return 1 - 2 * np.array(x)


_S, _C, _Z = _mseq((2, 0)), _mseq((3, 0)), _mseq((4, 2, 1, 0))


def sss_m0_m1(n_id1):
    q1 = n_id1 // 30
    q = (n_id1 + q1 * (q1 + 1) // 2) // 30
    mp = n_id1 + q * (q + 1) // 2
    m0 = mp % 31
    return m0, (m0 + mp // 31 + 1) % 31


@lru_cache(maxsize=None)
def sss_seq(n_id1, n_id2, subframe):
    m0, m1 = sss_m0_m1(n_id1)
    n = np.arange(31)
    s0, s1 = _S[(n + m0) % 31], _S[(n + m1) % 31]
    c0, c1 = _C[(n + n_id2) % 31], _C[(n + n_id2 + 3) % 31]
    z0, z1 = _Z[(n + m0 % 8) % 31], _Z[(n + m1 % 8) % 31]
    d = np.empty(62)
    if subframe == 0:
        d[0::2], d[1::2] = s0 * c0, s1 * c1 * z0
    else:
        d[0::2], d[1::2] = s1 * c0, s0 * c1 * z1
    return d


def crs_values(pci, ns, l):
    c = gold(2 ** 10 * (7 * (ns + 1) + l + 1) * (2 * pci + 1) + 2 * pci + 1, 440)
    m = np.arange(2 * N_RB) + 110 - N_RB
    return ((1 - 2.0 * c[2 * m]) + 1j * (1 - 2.0 * c[2 * m + 1])) / np.sqrt(2)


def crs_k(pci, port, l, ns):
    """CRS subcarriers of `port` in symbol l (0..6) of slot ns; empty if none."""
    if l not in ((0, 4) if port < 2 else (1,)):
        return np.empty(0, int)
    v = {0: 0 if l == 0 else 3, 1: 3 if l == 0 else 0, 2: 3 * (ns % 2), 3: 3 + 3 * (ns % 2)}[port]
    return 6 * np.arange(2 * N_RB) + (v + pci % 6) % 6


def crs_k_ports(pci, n_ports, l, ns):
    ks = [crs_k(pci, p, l, ns) for p in range(n_ports)]
    return np.unique(np.concatenate(ks)) if ks else np.empty(0, int)


# ================================================================== bit-level tools (36.212)

def crc(bits, kind):
    """CRC bits (MSB first) of a 0/1 array. kind: '16', '24A', '24B'."""
    poly, width = {"16": (0x1021, 16), "24A": (0x864CFB, 24), "24B": (0x800063, 24)}[kind]
    reg, top = 0, 1 << width
    for b in np.concatenate([np.asarray(bits, np.uint8), np.zeros(width, np.uint8)]):
        reg = (reg << 1) | int(b)
        if reg & top:
            reg ^= poly
    return np.array([(reg >> (width - 1 - i)) & 1 for i in range(width)], np.uint8)


def crc_ok(bits_with_crc, kind):
    w = 16 if kind == "16" else 24
    return bool(np.array_equal(crc(bits_with_crc[:-w], kind), bits_with_crc[-w:]))


def to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


# ---- tail-biting convolutional code (PBCH, PDCCH), rate 1/3, K = 7

_G = [np.array([(g >> (6 - j)) & 1 for j in range(7)]) for g in (0o133, 0o171, 0o165)]
_NEXT_C = np.zeros((64, 2), int)
_OUT_C = np.zeros((64, 2, 3), np.uint8)
for _s in range(64):
    _hist = [(_s >> j) & 1 for j in range(6)]
    for _b in (0, 1):
        _OUT_C[_s, _b] = [int(np.dot(g, [_b] + _hist)) % 2 for g in _G]
        _NEXT_C[_s, _b] = ((_s << 1) | _b) & 63
_P0, _P1, _INB = np.arange(64) >> 1, (np.arange(64) >> 1) | 32, np.arange(64) & 1


def conv_encode(bits):
    n, state = len(bits), 0
    for j in range(6):
        state |= int(bits[n - 1 - j]) << j
    d = np.zeros((3, n), np.uint8)
    for k in range(n):
        d[:, k] = _OUT_C[state, int(bits[k])]
        state = _NEXT_C[state, int(bits[k])]
    return d


def viterbi(soft):
    """Tail-biting Viterbi on soft streams (3, n); positive soft value = bit 0."""
    n = soft.shape[1]
    expected = 1.0 - 2.0 * _OUT_C
    metric = np.full((64, 64), -np.inf)
    metric[np.arange(64), np.arange(64)] = 0.0
    dec = np.zeros((n, 64, 64), bool)
    for t in range(n):
        bm = expected @ soft[:, t]
        c0 = metric[:, _P0] + bm[_P0, _INB][None, :]
        c1 = metric[:, _P1] + bm[_P1, _INB][None, :]
        dec[t] = c1 > c0
        metric = np.where(dec[t], c1, c0)
    start = int(np.argmax(metric[np.arange(64), np.arange(64)]))
    state, out = start, np.zeros(n, np.uint8)
    for t in range(n - 1, -1, -1):
        out[t] = state & 1
        state = int(_P1[state] if dec[t, start, state] else _P0[state])
    return out


# ---- sub-block interleaver and rate matching (36.212 5.1.4)

_PERM_TURBO = np.array([0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
                        1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31])
_PERM_CONV = np.array([1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31,
                       0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30])


def subblock_indices(d_len, shifted=False, turbo=False):
    """idx[j] = input position landing at output j, or -1 for a NULL."""
    perm = _PERM_TURBO if turbo else _PERM_CONV
    rows = -(-d_len // 32)
    k_pi = rows * 32
    padded = np.arange(k_pi) - (k_pi - d_len)
    padded[padded < 0] = -1
    if not shifted:
        return padded.reshape(rows, 32)[:, perm].T.reshape(-1)
    k = np.arange(k_pi)
    return padded[(perm[k // rows] + 32 * (k % rows) + 1) % k_pi]


def _take(stream, idx):
    out = np.full(len(idx), -1)
    ok = idx >= 0
    out[ok] = stream[idx[ok]]
    return out


def derate_conv(soft, d_len):
    """Soft bits -> soft streams (3, d_len) for a convolutionally coded block."""
    idx = subblock_indices(d_len)
    ids = np.arange(3 * d_len).reshape(3, d_len)
    w = np.concatenate([_take(ids[i], idx) for i in range(3)])
    valid = np.flatnonzero(w >= 0)
    pos = np.tile(valid, len(soft) // len(valid) + 1)[:len(soft)]
    out = np.zeros(3 * d_len)
    np.add.at(out, w[pos], soft)
    return out.reshape(3, d_len)


def derate_turbo(soft, k, n_filler, rv):
    """Soft bits of one code block -> soft streams (3, K+4). Filler bits are NULL in
    d0 and d1 (skipped by the transmitter) and known to be 0."""
    d_len = k + 4
    ids = np.arange(3 * d_len).reshape(3, d_len)
    ids[0, :n_filler] = -1
    ids[1, :n_filler] = -1
    idx = subblock_indices(d_len, turbo=True)
    idx2 = subblock_indices(d_len, shifted=True, turbo=True)
    v0, v1, v2 = _take(ids[0], idx), _take(ids[1], idx), _take(ids[2], idx2)
    w = np.concatenate([v0, np.stack([v1, v2], axis=1).reshape(-1)])
    rows = len(idx) // 32
    k0 = rows * (2 * int(np.ceil(len(w) / (8 * rows))) * rv + 2)
    valid = np.flatnonzero(w >= 0)
    order = np.roll(valid, -np.searchsorted(valid, k0))
    pos = np.tile(order, len(soft) // len(order) + 1)[:len(soft)]
    out = np.zeros(3 * d_len)
    np.add.at(out, w[pos], soft)
    out = out.reshape(3, d_len)
    out[0, :n_filler] = out[1, :n_filler] = 1e3
    return out


# ---- turbo code (36.212 5.1.3.2), max-log-MAP decoder in plain numpy

_NEXT_T = np.zeros((8, 2), int)
_PAR_T = np.zeros((8, 2), int)
for _s in range(8):
    for _u in (0, 1):
        _s1, _s2, _s3 = _s & 1, (_s >> 1) & 1, (_s >> 2) & 1
        _a = _u ^ _s2 ^ _s3
        _PAR_T[_s, _u] = _a ^ _s1 ^ _s3
        _NEXT_T[_s, _u] = _a | (_s1 << 1) | (_s2 << 2)
_PREV = [[(s, u) for s in range(8) for u in (0, 1) if _NEXT_T[s, u] == ns] for ns in range(8)]
_PS = np.array([[p[0] for p in pr] for pr in _PREV])
_PU = np.array([[p[1] for p in pr] for pr in _PREV])
_SU = np.array([[1.0, -1.0]] * 8)
_SP = 1.0 - 2.0 * _PAR_T


def _max_log_map(ls, lp, la, n_info):
    n = len(ls)
    la_full = np.concatenate([la, np.zeros(n - n_info)])
    g = 0.5 * (_SU[None] * (ls + la_full)[:, None, None] + _SP[None] * lp[:, None, None])
    alpha = np.full((n + 1, 8), -np.inf)
    alpha[0, 0] = 0.0
    for t in range(n):
        a = (alpha[t][_PS] + g[t][_PS, _PU]).max(axis=1)
        alpha[t + 1] = a - a.max()
    beta = np.full((n + 1, 8), -np.inf)
    beta[n, 0] = 0.0
    for t in range(n - 1, -1, -1):
        b = (beta[t + 1][_NEXT_T] + g[t]).max(axis=1)
        beta[t] = b - b.max()
    a, bn = alpha[:n_info], beta[1:n_info + 1]
    m0 = (a + g[:n_info, :, 0] + bn[:, _NEXT_T[:, 0]]).max(axis=1)
    m1 = (a + g[:n_info, :, 1] + bn[:, _NEXT_T[:, 1]]).max(axis=1)
    return (m0 - m1) - ls[:n_info] - la


def turbo_decode(soft, k, check, n_iter=8, scale=0.75):
    """Soft streams (3, K+4) -> (bits, ok). Stops as soon as check(bits) passes."""
    i = TURBO_K.index(k)
    pi = (TURBO_F1[i] * np.arange(k, dtype=np.int64) + TURBO_F2[i] * np.arange(k, dtype=np.int64) ** 2) % k
    inv = np.empty(k, int)
    inv[pi] = np.arange(k)
    x = np.concatenate([soft[0, :k], [soft[0, k], soft[2, k], soft[1, k + 1]]])
    z = np.concatenate([soft[1, :k], [soft[1, k], soft[0, k + 1], soft[2, k + 1]]])
    xp = np.concatenate([soft[0, :k][pi], [soft[0, k + 2], soft[2, k + 2], soft[1, k + 3]]])
    zp = np.concatenate([soft[2, :k], [soft[1, k + 2], soft[0, k + 3], soft[2, k + 3]]])
    ext2 = np.zeros(k)
    bits = (soft[0, :k] < 0).astype(np.uint8)
    for _ in range(n_iter):
        ext1 = scale * _max_log_map(x, z, ext2, k)
        ext2 = (scale * _max_log_map(xp, zp, ext1[pi], k))[inv]
        bits = (soft[0, :k] + ext1 + ext2 < 0).astype(np.uint8)
        if check(bits):
            return bits, True
    return bits, False


# ================================================================== OFDM and channel

def ofdm_demod(x, start):
    """One subframe starting at sample `start` -> grid (14, N_SC). FFT window starts half
    way into the CP; the resulting phase ramp is removed."""
    grid = np.empty((14, N_SC), complex)
    k = np.arange(N_FFT)
    pos = start
    for l in range(14):
        cp = CP[l % 7]
        off = cp // 2
        spec = np.fft.fft(x[pos + cp - off:pos + cp - off + N_FFT]) * np.sqrt(N_SC) / N_FFT
        grid[l] = (spec * np.exp(2j * np.pi * k * off / N_FFT))[BINS]
        pos += cp + N_FFT
    return grid


def channel_estimate(y, pci, subframe, port):
    """CRS least-squares estimate, smoothed over 3 CRS, interpolated over the grid.
    Returns (h (14, N_SC), noise variance per RE)."""
    syms, hs, res = [], [], []
    for l in range(14):
        ns, ls = 2 * subframe + l // 7, l % 7
        ks = crs_k(pci, port, ls, ns)
        if not len(ks):
            continue
        est = y[l, ks] * np.conj(crs_values(pci, ns, ls))
        sm = np.convolve(est, np.ones(3) / 3, mode="same")
        sm[0], sm[-1] = est[:2].mean(), est[-2:].mean()
        res.append(est - sm)
        hs.append(np.interp(np.arange(N_SC), ks, sm.real) + 1j * np.interp(np.arange(N_SC), ks, sm.imag))
        syms.append(l)
    hs = np.array(hs)
    h = np.empty((14, N_SC), complex)
    for k in range(N_SC):
        h[:, k] = np.interp(np.arange(14), syms, hs[:, k].real) + 1j * np.interp(np.arange(14), syms, hs[:, k].imag)
    return h, float(np.mean(np.abs(np.concatenate(res)) ** 2) / (1 - 1 / 3))


def combine(rx, h, n_ports, nv):
    """Equalise REs: 1 port, or 2-port SFBC. Returns (symbols, csi = |h|^2 / noise)."""
    if n_ports == 1:
        g = np.abs(h[0]) ** 2
        return np.conj(h[0]) * rx / np.maximum(g, 1e-12), g / nv
    y0, y1, h0, h1 = rx[0::2], rx[1::2], h[0, 0::2], h[1, 0::2]
    g = np.maximum(np.abs(h0) ** 2 + np.abs(h1) ** 2, 1e-12)
    sym = np.empty(len(rx), complex)
    sym[0::2] = (np.conj(h0) * y0 + h1 * np.conj(y1)) / g * np.sqrt(2)
    sym[1::2] = (np.conj(h0) * y1 - h1 * np.conj(y0)) / g * np.sqrt(2)
    return sym, np.repeat(g / 2, 2) / nv


def _pam(qm):
    """Per-dimension amplitude levels and the bits each level carries (36.211 7.1)."""
    nb = qm // 2
    bits = np.array([[(v >> (nb - 1 - j)) & 1 for j in range(nb)] for v in range(2 ** nb)])
    s = 1 - 2 * bits
    if qm == 2:
        lv = s[:, 0] / np.sqrt(2)
    elif qm == 4:
        lv = s[:, 0] * (2 - s[:, 1]) / np.sqrt(10)
    else:
        lv = s[:, 0] * (4 - s[:, 1] * (2 - s[:, 2])) / np.sqrt(42)
    return lv, bits


def qam_soft(sym, csi, qm):
    """Max-log soft bits (positive = bit 0) for QPSK/16QAM/64QAM symbols."""
    lv, bits = _pam(qm)
    out = np.empty((len(sym), qm))
    for dim, v in ((0, sym.real), (1, sym.imag)):
        d = (v[:, None] - lv[None, :]) ** 2
        for j in range(qm // 2):
            out[:, 2 * j + dim] = (d[:, bits[:, j] == 1].min(1) - d[:, bits[:, j] == 0].min(1)) * csi
    return out.reshape(-1)


def descramble(soft, c_init):
    return soft * (1.0 - 2.0 * gold(c_init, len(soft)))


# ================================================================== 1-2. cell search

def pss_time(n_id2):
    s = np.zeros(N_FFT, complex)
    s[BINS[SYNC62]] = pss_seq(n_id2)
    return np.fft.ifft(s)


def find_pss(x, max_cfo=60e3, step=2e3):
    """Search the 3 PSS over a grid of frequency offsets. Returns (n_id2, cfo, position)."""
    t = np.arange(len(x)) / FS
    n = len(x) + N_FFT
    refs = [np.conj(np.fft.fft(pss_time(i), n)) for i in range(3)]
    best = (0, 0, 0.0, 0)
    for cfo in np.arange(-max_cfo, max_cfo + 1, step):
        spec = np.fft.fft(x * np.exp(-2j * np.pi * cfo * t), n)
        for n_id2, ref in enumerate(refs):
            c = np.abs(np.fft.ifft(spec * ref)[:len(x) - N_FFT])
            if c.max() > best[0]:
                best = (c.max(), n_id2, cfo, int(c.argmax()))
    return best[1:]


def cp_cfo(x, first_symbol):
    """Fine frequency offset from the cyclic prefix of every symbol (+/- 7.5 kHz range)."""
    acc, pos, l = 0j, first_symbol, 0
    while pos + CP[l % 7] + N_FFT <= len(x):
        cp = CP[l % 7]
        acc += np.vdot(x[pos:pos + cp], x[pos + N_FFT:pos + N_FFT + cp])
        pos += cp + N_FFT
        l += 1
    return float(np.angle(acc) / (2 * np.pi) * FS / N_FFT)


# SSS position relative to the PSS (in samples, body to body) for each frame structure.
SSS_OFFSETS = {("FDD", "normal"): 36 + 512, ("FDD", "extended"): 128 + 512,
               ("TDD", "normal"): 36 + 548 + 552 + 512, ("TDD", "extended"): 3 * 640}


def detect_sss(x, n_id2, pss_pos):
    """Try every frame structure and all 168 x 2 SSS. Returns the best match."""
    y_pss = np.fft.fft(x[pss_pos:pss_pos + N_FFT])[BINS[SYNC62]]
    h = np.convolve(y_pss * np.conj(pss_seq(n_id2)), np.ones(3) / 3, mode="same")
    best = None
    for (duplex, cp), off in SSS_OFFSETS.items():
        y_sss = np.fft.fft(x[pss_pos - off:pss_pos - off + N_FFT])[BINS[SYNC62]]
        eq = y_sss * np.conj(h)
        for n_id1 in range(168):
            for sf in (0, 5):
                m = float(np.real(np.sum(eq * sss_seq(n_id1, n_id2, sf)))) / (np.sum(np.abs(eq)) + 1e-12)
                if best is None or m > best["metric"]:
                    best = {"metric": m, "n_id1": n_id1, "subframe": sf, "duplex": duplex, "cp": cp}
    return best


# ================================================================== 3. PBCH / MIB

@lru_cache(maxsize=None)
def pbch_re(pci):
    """240 (l, k): subframe 0 symbols 7..10, centre 72 subcarriers, minus CRS of 4 ports."""
    out = [(l, k) for l in range(7, 11) for k in CENTER72
           if k not in set(crs_k_ports(pci, 4, l - 7, 1).tolist())]
    return np.array(out)


PBCH_MASKS = {1: np.zeros(16, np.uint8), 2: np.ones(16, np.uint8), 4: np.array([0, 1] * 8, np.uint8)}
BANDWIDTHS = [6, 15, 25, 50, 75, 100]
PHICH_NG = ["1/6", "1/2", "1", "2"]


def decode_mib(y, h01, nv, pci):
    """Blind PBCH decode of one frame. Tries 1- and 2-port reception, the 4 positions in
    the 40 ms period and the 3 CRC masks (which encode the number of antenna ports)."""
    pos = pbch_re(pci)
    rx = y[pos[:, 0], pos[:, 1]]
    h = h01[:, pos[:, 0], pos[:, 1]]
    best = None
    for rx_ports in (1, 2):
        sym, csi = combine(rx, h[:rx_ports], rx_ports, nv)
        soft = qam_soft(sym, csi, 2)
        for i in range(4):
            full = np.zeros(1920)
            full[480 * i:480 * (i + 1)] = soft * (1.0 - 2.0 * gold(pci, 1920)[480 * i:480 * (i + 1)])
            streams = derate_conv(full, 40)
            bits = viterbi(streams)
            for ports, mask in PBCH_MASKS.items():
                cand = bits.copy()
                cand[24:] ^= mask
                if crc_ok(cand, "16"):
                    metric = float(np.sum(streams * (1.0 - 2.0 * conv_encode(bits))))
                    if best is None or metric > best["metric"]:
                        best = {"bits": cand[:24], "ports": ports, "frame_in_period": i, "metric": metric}
    if best is None:
        return None
    b = best["bits"]
    return {"n_rb": BANDWIDTHS[to_int(b[0:3])], "phich_duration": ["normal", "extended"][b[3]],
            "phich_ng": PHICH_NG[to_int(b[4:6])], "sfn": (to_int(b[6:14]) << 2) | best["frame_in_period"],
            "ports": best["ports"], "spare": to_int(b[14:24])}


# ================================================================== 4-5. control region

def _regs(pci, n_ports, l):
    """REGs of control symbol l as (lowest k, [k...]) in increasing k (36.211 6.2.4)."""
    regs = []
    if l == 0 or (l == 1 and n_ports == 4):
        crs = set(crs_k_ports(pci, max(n_ports, 2) if l == 0 else 4, l, 0).tolist())
        for base in range(0, N_SC, 6):
            regs.append((base, [k for k in range(base, base + 6) if k not in crs]))
    else:
        for base in range(0, N_SC, 4):
            regs.append((base, list(range(base, base + 4))))
    return regs


@lru_cache(maxsize=None)
def control_region(pci, n_ports, cfi, n_phich_groups):
    """Positions of PCFICH (4 REGs), PHICH and PDCCH REGs (in PDCCH mapping order)."""
    per = {l: _regs(pci, n_ports, l) for l in range(cfi)}
    used = {l: [False] * len(per[l]) for l in range(cfi)}
    index0 = {reg[0]: i for i, reg in enumerate(per[0])}
    pcfich = []
    for i in range(4):
        j = index0[(6 * (pci % (2 * N_RB)) + (i * N_RB // 2) * 6) % N_SC]
        used[0][j] = True
        pcfich.append(per[0][j][1])
    free0 = [i for i, u in enumerate(used[0]) if not u]
    n0 = len(free0)
    for m in range(n_phich_groups):
        for i in range(3):
            used[0][free0[(pci + m + i * n0 // 3) % n0]] = True
    lookup = {l: {reg[0]: i for i, reg in enumerate(per[l])} for l in range(cfi)}
    pdcch = []
    for k in range(N_SC):
        for l in range(cfi):
            i = lookup[l].get(k)
            if i is not None and not used[l][i]:
                pdcch.append([(l, kk) for kk in per[l][i][1]])
    return np.array([[(0, k) for k in ks] for ks in pcfich]), np.array(pdcch)


CFI_CODEWORDS = {cfi: np.array(([0, 1, 1], [1, 0, 1], [1, 1, 0])[cfi - 1] * 11, np.uint8)[:32] for cfi in (1, 2, 3)}


def decode_cfi(y, h, nv, pci, n_ports, subframe, n_phich):
    pos = control_region(pci, n_ports, 1, n_phich)[0].reshape(-1, 2)
    sym, csi = combine(y[pos[:, 0], pos[:, 1]], h[:, pos[:, 0], pos[:, 1]], n_ports, nv)
    soft = descramble(qam_soft(sym, csi, 2), (subframe + 1) * (2 * pci + 1) * 2 ** 9 + pci)
    scores = {c: float(np.dot(soft, 1.0 - 2.0 * w)) for c, w in CFI_CODEWORDS.items()}
    return max(scores, key=scores.get)


def pdcch_soft(y, h, nv, pci, n_ports, subframe, cfi, n_phich):
    """PDCCH REs -> descrambled soft bits in CCE order (undoes REG interleaving and shift)."""
    regs = control_region(pci, n_ports, cfi, n_phich)[1]
    n_quad = len(regs)
    idx = subblock_indices(n_quad)
    order = np.roll(idx[idx >= 0], -(pci % n_quad))      # REG i carries quadruplet order[i]
    pos = regs.reshape(-1, 2)
    rx, hh = y[pos[:, 0], pos[:, 1]], h[:, pos[:, 0], pos[:, 1]]
    rx_seq, h_seq = np.empty_like(rx), np.empty_like(hh)
    for i, q in enumerate(order):
        rx_seq[4 * q:4 * q + 4] = rx[4 * i:4 * i + 4]
        h_seq[:, 4 * q:4 * q + 4] = hh[:, 4 * i:4 * i + 4]
    n_cce = n_quad // 9
    sym, csi = combine(rx_seq[:n_cce * 36], h_seq[:, :n_cce * 36], n_ports, nv)
    return descramble(qam_soft(sym, csi, 2), subframe * 2 ** 9 + pci), n_cce


AMBIGUOUS = {12, 14, 16, 20, 24, 26, 32, 40, 44, 56}
RIV_BITS = int(np.ceil(np.log2(N_RB * (N_RB + 1) / 2)))
RBG = 2                                                  # resource block group size for 25 RB
SIZE_1A = 1 + 1 + RIV_BITS + 5 + 3 + 1 + 2 + 2           # FDD, same size as format 0 after padding
SIZE_1A += SIZE_1A in AMBIGUOUS
SIZE_1 = 1 + -(-N_RB // RBG) + 5 + 3 + 1 + 2 + 2
while SIZE_1 == SIZE_1A or SIZE_1 in AMBIGUOUS:
    SIZE_1 += 1


def find_dcis(soft, n_cce):
    """Try every CCE position and aggregation level for DCI formats 1A and 1. The CRC is
    scrambled with the RNTI, so for a correct decode CRC(payload) XOR received CRC = RNTI.
    A decode is accepted when re-encoding it matches the received soft bits closely."""
    found = {}
    for agg in (8, 4, 2, 1):
        for first in range(0, n_cce - agg + 1, agg):
            seg = soft[first * 72:(first + agg) * 72]
            for size in (SIZE_1A, SIZE_1):
                streams = derate_conv(seg, size + 16)
                bits = viterbi(streams)
                quality = float(np.sum(streams * (1.0 - 2.0 * conv_encode(bits))) / (np.sum(np.abs(streams)) + 1e-12))
                if quality < 0.9:
                    continue
                rnti = to_int(crc(bits[:size], "16") ^ bits[size:])
                key = (size, rnti, bits[:size].tobytes())
                if key not in found:
                    found[key] = {"size": size, "rnti": rnti, "bits": bits[:size], "agg": agg,
                                  "cce": first, "quality": quality}
    return list(found.values())


def mcs_to_qm_itbs(i_mcs):
    """36.213 Table 7.1.7.1-1."""
    if i_mcs <= 9:
        return 2, i_mcs
    if i_mcs <= 16:
        return 4, i_mcs - 1
    if i_mcs <= 28:
        return 6, i_mcs - 2
    return {29: 2, 30: 4, 31: 6}[i_mcs], None


def parse_dci(d):
    """Fields of DCI format 1A or 1 (FDD, 25 RB)."""
    b = d["bits"]
    if d["size"] == SIZE_1A:
        if b[0] == 0:
            return {"format": "0 (uplink grant)"}
        riv = to_int(b[2:2 + RIV_BITS])
        length, start = riv // N_RB + 1, riv % N_RB
        if length - 1 > N_RB // 2 or start + length > N_RB:
            length, start = N_RB - riv // N_RB + 1, N_RB - 1 - riv % N_RB
        p = 2 + RIV_BITS
        out = {"format": "1A", "prbs": list(range(start, start + length)), "mcs": to_int(b[p:p + 5]),
               "harq": to_int(b[p + 5:p + 8]), "ndi": int(b[p + 8]), "rv": to_int(b[p + 9:p + 11]),
               "tpc": to_int(b[p + 11:p + 13])}
        if d["rnti"] >= 0xFFF4:   # SI/P/RA-RNTI: QPSK, I_TBS = MCS field, N_PRB from TPC bit
            out.update(qm=2, i_tbs=out["mcs"], n_prb_tbs=3 if out["tpc"] & 1 else 2)
            return out
    else:
        n_rbg = -(-N_RB // RBG)
        if b[0] == 1:
            return {"format": "1 (allocation type 1, not decoded)"}
        bitmap = b[1:1 + n_rbg]
        prbs = [rb for g in range(n_rbg) if bitmap[g] for rb in range(RBG * g, min(RBG * g + RBG, N_RB))]
        p = 1 + n_rbg
        out = {"format": "1", "prbs": prbs, "mcs": to_int(b[p:p + 5]), "harq": to_int(b[p + 5:p + 8]),
               "ndi": int(b[p + 8]), "rv": to_int(b[p + 9:p + 11]), "tpc": to_int(b[p + 11:p + 13])}
    out["qm"], out["i_tbs"] = mcs_to_qm_itbs(out["mcs"])
    out["n_prb_tbs"] = len(out["prbs"])
    return out


# ================================================================== 6. PDSCH

def pdsch_re(pci, n_ports, subframe, cfi, prbs):
    """(l, k) of the PDSCH in mapping order: subcarrier first, then symbol."""
    ks = np.concatenate([np.arange(12 * p, 12 * p + 12) for p in sorted(prbs)])
    avoid = set()
    if subframe in (0, 5):
        avoid |= {(l, int(k)) for l in (5, 6) for k in CENTER72}
    if subframe == 0:
        avoid |= {(l, int(k)) for l in range(7, 11) for k in CENTER72}
    out = []
    for l in range(cfi, 14):
        crs = set(crs_k_ports(pci, n_ports, l % 7, 2 * subframe + l // 7).tolist())
        out += [(l, int(k)) for k in ks if int(k) not in crs and (l, int(k)) not in avoid]
    return np.array(out)


def segment(tbs):
    """Code block segmentation (36.212 5.1.2) -> (list of K per block, filler bits)."""
    b = tbs + 24
    if b <= 6144:
        c, b2 = 1, b
    else:
        c = -(-b // (6144 - 24))
        b2 = b + 24 * c
    k_plus = min(k for k in TURBO_K if c * k >= b2)
    if c == 1:
        ks = [k_plus]
    else:
        k_minus = max(k for k in TURBO_K if k < k_plus)
        c_minus = (c * k_plus - b2) // (k_plus - k_minus)
        ks = [k_minus] * c_minus + [k_plus] * (c - c_minus)
    return ks, sum(ks) - b2


def decode_pdsch(y, h, nv, pci, n_ports, subframe, cfi, dci):
    """Returns (transport block bits or None, details)."""
    re = pdsch_re(pci, n_ports, subframe, cfi, dci["prbs"])
    sym, csi = combine(y[re[:, 0], re[:, 1]], h[:, re[:, 0], re[:, 1]], n_ports, nv)
    qm = dci["qm"]
    soft = descramble(qam_soft(sym, csi, qm), dci["rnti"] * 2 ** 14 + subframe * 2 ** 9 + pci)
    tbs = TBS_TABLE[dci["i_tbs"]][dci["n_prb_tbs"] - 1]
    ks, n_filler = segment(tbs)
    c = len(ks)
    g_prime = len(re)                                  # G / (N_L * Qm) with one layer
    gamma = g_prime % c
    pos, out = 0, []
    for r, k in enumerate(ks):
        e = qm * (g_prime // c if r <= c - gamma - 1 else -(-g_prime // c))
        f = n_filler if r == 0 else 0
        streams = derate_turbo(soft[pos:pos + e], k, f, dci["rv"])
        pos += e
        check = (lambda bb: crc_ok(bb, "24B")) if c > 1 else (lambda bb, f=f: crc_ok(bb[f:], "24A"))
        bits, ok = turbo_decode(streams, k, check)
        if not ok:
            return None, {"tbs": tbs, "blocks": c, "failed_block": r}
        out.append(bits[f:k - (24 if c > 1 else 0)])
    tb = np.concatenate(out)
    return (tb[:tbs] if crc_ok(tb, "24A") else None), {"tbs": tbs, "blocks": c}


def repeating_pattern(bits, max_period=64):
    for p in range(1, max_period + 1):
        if np.array_equal(bits[p:], bits[:-p]):
            return bits[:p]
    return None


# ================================================================== main

def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else FILE)
    x = np.fromfile(path, dtype=np.complex64).astype(complex)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    fs = meta.get("sample_rate_hz", FS)
    # Say exactly which recording this is, so an old file is never mistaken for a new one.
    made = time.localtime(path.stat().st_mtime)
    age_min = (time.time() - path.stat().st_mtime) / 60
    print("File      %s" % path.resolve())
    print("Recorded  %s  (%s ago)" % (time.strftime("%Y-%m-%d %H:%M:%S", made),
                                    "%.0f s" % (age_min * 60) if age_min < 1 else "%.0f min" % age_min))
    if meta.get("center_frequency_hz"):
        print("Frequency %.3f MHz" % (meta["center_frequency_hz"] / 1e6))
    if age_min > 2:
        print("WARNING   this recording is %.0f minutes old -- did lte_send.py finish without an error?" % age_min)
    print()
    if abs(fs - FS) > 1:
        sys.exit("needs 5 MHz LTE at 7.68 Msps (file is %.2f Msps)" % (fs / 1e6))
    need = (N_FRAMES + 1) * FRAME
    if len(x) < need:                         # a short file (e.g. 10 ms from MATLAB) loops,
        print("%s is %.1f ms long: looping it, like the Pluto does" % (path.name, 1e3 * len(x) / FS))
        x = np.tile(x, -(-need // len(x)))
        start = 0
    else:
        start = min(int(START_S * FS), len(x) - need)
    x = x[start:start + need]
    x = x - x.mean()      # remove the receiver's DC offset (HackRF LO leakage) before anything else
    print("Decoding %s from %.2f s, %d frames\n" % (path.name, start / FS, N_FRAMES))

    # ---- 1. PSS
    n_id2, cfo, _ = find_pss(x[:FRAME // 2 + N_FFT])
    x = x * np.exp(-2j * np.pi * cfo * np.arange(len(x)) / FS)
    corr = np.abs(np.correlate(x[4000:4000 + FRAME // 2 + N_FFT], pss_time(n_id2), "valid"))
    pss_pos = 4000 + int(corr.argmax())
    slot_start = pss_pos - CP[6] - SYM_START[6]
    fine = cp_cfo(x, slot_start % SLOT)
    x = x * np.exp(-2j * np.pi * fine * np.arange(len(x)) / FS)
    cfo += fine

    # ---- 2. SSS
    sss = detect_sss(x, n_id2, pss_pos)
    pci = 3 * sss["n_id1"] + n_id2
    print("CELL SEARCH (PSS + SSS)")
    print("  cell ID (PCI)       %d   (N_ID1 %d, N_ID2 %d)" % (pci, sss["n_id1"], n_id2))
    print("  duplex mode         %s" % sss["duplex"])
    print("  cyclic prefix       %s" % sss["cp"])
    print("  frequency offset    %+.2f kHz" % (cfo / 1e3))
    print("  SSS match           %.2f  (1.0 = perfect)" % sss["metric"])
    if (sss["duplex"], sss["cp"]) != ("FDD", "normal"):
        sys.exit("only FDD with normal CP is decoded further")
    frame0 = (slot_start - (10 * SLOT if sss["subframe"] == 5 else 0)) % FRAME
    frames = [frame0 + i * FRAME for i in range(N_FRAMES) if frame0 + (i + 1) * FRAME <= len(x)]
    same = [abs(np.vdot(x[frames[0]:frames[0] + FRAME], x[f:f + FRAME])) /
            (np.linalg.norm(x[frames[0]:frames[0] + FRAME]) * np.linalg.norm(x[f:f + FRAME])) for f in frames[1:]]
    period = next((i + 1 for i, s in enumerate(same) if s > 0.9), None)
    print("  signal repeats      %s" % ("every %d ms" % (10 * period) if period else
                                         "not seen (live signal, or radio clock drift hides it)"))

    # ---- 3. MIB from every frame
    mibs = []
    for f in frames:
        y = ofdm_demod(x, f)
        h0, nv = channel_estimate(y, pci, 0, 0)
        h1, _ = channel_estimate(y, pci, 0, 1)
        m = decode_mib(y, np.array([h0, h1]), nv, pci)
        mibs.append(m)
    m = next((m for m in mibs if m), None)
    print("\nMIB (PBCH)")
    if m is None:
        sys.exit("  could not decode the MIB")
    print("  bandwidth           %d RB (%.1f MHz)" % (m["n_rb"], {6: 1.4, 15: 3, 25: 5, 50: 10, 75: 15, 100: 20}[m["n_rb"]]))
    print("  antenna ports       %d" % m["ports"])
    print("  PHICH duration      %s" % m["phich_duration"])
    print("  PHICH Ng            %s" % m["phich_ng"])
    print("  frame number (SFN)  %s" % ", ".join(str(mm["sfn"]) if mm else "?" for mm in mibs))
    print("  decoded in          %d of %d frames" % (sum(1 for mm in mibs if mm), len(mibs)))
    n_ports = m["ports"]
    if m["n_rb"] != N_RB or n_ports > 2:
        sys.exit("this script decodes further only for 25 RB with 1 or 2 ports")
    n_phich = int(np.ceil({"1/6": 1 / 6, "1/2": 0.5, "1": 1, "2": 2}[m["phich_ng"]] * N_RB / 8))

    # ---- 4-5. control region of every subframe of one frame: CFI and all DCIs. Use a frame
    # where a broadcast SIB1 would be sent (even SFN; SFN divisible by 8 carries its first copy).
    pick = next((i for i, mm in enumerate(mibs) if mm and mm["sfn"] % 8 == 0),
                next((i for i, mm in enumerate(mibs) if mm and mm["sfn"] % 2 == 0), 0))
    frame_sf = frames[pick]
    subframes = []
    for sf in range(10):
        y = ofdm_demod(x, frame_sf + sf * SUBFRAME)
        hs = [channel_estimate(y, pci, sf, p) for p in range(n_ports)]
        h, nv = np.array([a for a, _ in hs]), float(np.mean([b for _, b in hs]))
        cfi = decode_cfi(y, h, nv, pci, n_ports, sf, n_phich)
        soft, n_cce = pdcch_soft(y, h, nv, pci, n_ports, sf, cfi, n_phich)
        subframes.append({"sf": sf, "y": y, "h": h, "nv": nv, "cfi": cfi, "dcis": find_dcis(soft, n_cce)})

    # A real RNTI is used in many subframes; a decode of other control data gives a random
    # 16-bit "RNTI". With ~200 such decodes two can match by chance, three almost never
    # (~1 in 3000). Keep RNTIs seen in 3+ subframes, plus the broadcast RNTIs (SI 65535,
    # P 65534), and the best decode of each per subframe. A passing data CRC then confirms it.
    seen = {}
    for s_ in subframes:
        for r in {d["rnti"] for d in s_["dcis"]}:
            seen[r] = seen.get(r, 0) + 1
    real = {r for r, n in seen.items() if n >= 3 or r >= 0xFFFE}

    # ---- 6. data channel for every real DCI
    print("\nSUBFRAMES (frame with SFN %s)" % (mibs[pick]["sfn"] if mibs[pick] else "?"))
    print("  sf  CFI  DCI  RNTI   RBs  MCS  mod     HARQ  RV  NDI  TBS    blocks  data CRC")
    rows, tbs_bits = [], []
    for s_ in subframes:
        best = {}
        for d in s_["dcis"]:
            if d["rnti"] in real and (d["rnti"] not in best or d["quality"] > best[d["rnti"]]["quality"]):
                best[d["rnti"]] = d
        if not best:
            print("  %d   %d    -" % (s_["sf"], s_["cfi"]))
            rows.append({"sf": s_["sf"], "cfi": s_["cfi"]})
            continue
        for d in best.values():
            info = parse_dci(d)
            row = {"sf": s_["sf"], "cfi": s_["cfi"], "rnti": d["rnti"], **info}
            line = "  %d   %d    %-4s %-5d" % (s_["sf"], s_["cfi"], info["format"].split()[0], d["rnti"])
            if info.get("prbs") and max(info["prbs"]) < N_RB:
                line += "  %-3d  %-3d  %-6s  %-4d  %d   %d" % (
                    len(info["prbs"]), info["mcs"], {2: "QPSK", 4: "16QAM", 6: "64QAM"}[info["qm"]],
                    info["harq"], info["rv"], info["ndi"])
                # RV 0 decodes on its own. Other RVs need the UE's soft-buffer size, except
                # broadcast data (SI/P-RNTI), which has no limit, so those are decoded too.
                if info["i_tbs"] is not None and (info["rv"] == 0 or d["rnti"] >= 0xFFFE):
                    tb, det = decode_pdsch(s_["y"], s_["h"], s_["nv"], pci, n_ports, s_["sf"], s_["cfi"], row)
                    row.update(det, ok=tb is not None)
                    line += "    %-6d %-6d  %s" % (det["tbs"], det["blocks"], "PASS" if tb is not None else "fail")
                    if tb is not None:
                        tbs_bits.append(tb)
                        if d["rnti"] == 0xFFFF:
                            line += "   SIB1 bytes: " + np.packbits(tb).tobytes().hex()
                else:
                    line += "    (RV %d: needs earlier transmissions, skipped)" % info["rv"]
            print(line)
            rows.append(row)
    ignored = sum(len(s_["dcis"]) for s_ in subframes) - sum(1 for r in rows if "rnti" in r)
    print("  (%d one-off decodes of random control data ignored)" % ignored)

    # ---- power checks: PDSCH vs reference signals, and empty resource elements
    y = ofdm_demod(x, frame_sf + 1 * SUBFRAME)
    crs_pow = np.mean([np.mean(np.abs(y[l, crs_k(pci, 0, l % 7, 2 + l // 7)]) ** 2) for l in (0, 4, 7, 11)])
    data_syms = [l for l in range(3, 14) if l % 7 not in (0, 4)]
    data_pow = np.mean(np.abs(y[data_syms]) ** 2)
    y5 = ofdm_demod(x, frame_sf + 5 * SUBFRAME)
    empty = [l for l in range(3, 14) if l % 7 not in (0, 4, 5, 6)]
    empty_pow = np.mean(np.abs(y5[empty]) ** 2)

    print("\nSUMMARY")
    with_data = [r for r in rows if "prbs" in r]
    if with_data:
        rntis = sorted({r["rnti"] for r in with_data})
        print("  RNTI                %s" % ", ".join("%d (in %d subframes)" % (r, seen[r]) for r in rntis))
        print("  DCI format          %s" % ", ".join(sorted({r["format"] for r in with_data})))
        print("  CFI                 %s" % ", ".join(sorted({str(r["cfi"]) for r in rows})))
        print("  resource blocks     %s" % ", ".join(sorted({"%d (RB %d-%d)" % (len(r["prbs"]), r["prbs"][0], r["prbs"][-1]) for r in with_data})))
        print("  modulation          %s" % ", ".join(sorted({{2: "QPSK", 4: "16QAM", 6: "64QAM"}[r["qm"]] for r in with_data})))
        print("  HARQ processes      %d used (%s)" % (len({r["harq"] for r in with_data}), " ".join(str(r["harq"]) for r in with_data)))
        print("  RV per subframe     %s" % " ".join(str(r["rv"]) for r in with_data))
        print("  subframes with data %s" % " ".join(str(r["sf"]) for r in with_data))
    print("  PDSCH / ref. power  %+.1f dB   (MATLAB 'Rho' is the PDSCH power offset)" % (10 * np.log10(data_pow / crs_pow)))
    print("  empty REs           %+.1f dB vs reference signals (very low = empty, so no OCNG)"
          % (10 * np.log10(max(empty_pow, 1e-30) / crs_pow)))
    if tbs_bits:
        pattern = repeating_pattern(np.concatenate(tbs_bits))
        print("  data blocks passed  %d of %d tried" % (len(tbs_bits), sum(1 for r in rows if "ok" in r)))
        print("  data bits           %s" % ("repeating pattern " + " ".join(map(str, pattern)) if pattern is not None
                                            else "first 32: " + " ".join(map(str, tbs_bits[0][:32]))))

    # ---- signal level, like a phone reports it
    # RSRP = average power of one reference-signal subcarrier (15 kHz). A grid value g here is
    # sqrt(N_SC) x the subcarrier's amplitude in the time signal (see ofdm_demod), so the
    # subcarrier's power, as a fraction of the HackRF's full scale, is |g|^2 / N_SC.
    crs = [np.abs(s_["y"][l, crs_k(pci, 0, l % 7, 2 * s_["sf"] + l // 7)]) ** 2
           for s_ in subframes for l in (0, 4, 7, 11)]
    rsrp_db = 10 * np.log10(np.mean(np.concatenate(crs)) / N_SC)
    noise_db = 10 * np.log10(np.mean([s_["nv"] for s_ in subframes]) / N_SC)   # per 15 kHz
    print("\nSIGNAL LEVEL (per 15 kHz subcarrier)")
    cal_path = path.parent / "hackrf_cal.json"
    cal = json.loads(cal_path.read_text()) if cal_path.exists() else None
    # Gains (RF amp, LNA, VGA). Older files without them used amp off, LNA 24, VGA 20.
    gains = lambda d: (bool(d.get("hackrf_amp", False)), int(d.get("hackrf_lna", 24)) // 8 * 8,
                       int(d.get("hackrf_vga", 20)) // 2 * 2)
    if "pluto_atten_db" in meta:
        print("  settings            Pluto TX attenuation %g dB, HackRF amp %s / LNA %d / VGA %d"
              % ((-meta["pluto_atten_db"],) + (("on" if gains(meta)[0] else "off"),) + gains(meta)[1:]))
    freq_ok = cal and abs(cal["rx_freq_hz"] - meta.get("center_frequency_hz", cal["rx_freq_hz"])) < 1e6
    if freq_ok and gains(cal) == gains(meta):
        off = cal["offset_db"]
        print("  RSRP                %.1f dBm   (Starlink D2C phones: median about -121 dBm)" % (rsrp_db + off))
        print("  noise               %.1f dBm   -> receiver noise figure about %.0f dB"
              % (noise_db + off, noise_db + off + 174 - 10 * np.log10(15e3)))
        print("  SNR                 %.1f dB" % (rsrp_db - noise_db))
        print("  (calibration from %s)" % cal["made"])
    else:
        why = "no calibration yet" if not cal else ("calibrated at a different frequency" if not freq_ok
                                                    else "calibrated with different HackRF gains")
        print("  RSRP                %.1f dB full scale (%s -- run calibrate_hackrf.py with the same settings)"
              % (rsrp_db, why))
        print("  SNR                 %.1f dB" % (rsrp_db - noise_db))


if __name__ == "__main__":
    main()
