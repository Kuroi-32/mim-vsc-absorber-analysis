"""
chirality_sweep.py  (v2 — single stacked file, rotation sweep)
===============================================================
Parses a single CST parametric sweep export containing all 4 S-params
across all rotation values, and produces:

  1. CD_abs spectra overlaid across all rotation values
  2. Trade-off curve: peak CD_abs vs peak absorption (A_x, A_y)
  3. Linear absorption overlay across rotation values

Works for BOTH circular and linear port exports from the same file
structure — just point CIRC_FILE and LIN_FILE at the respective exports.

FILE STRUCTURE (as exported by CST parametric sweep):
  All rotation values for SZmax(1),Zmax(1) stacked first,
  then all rotation values for SZmax(2),Zmax(1), etc.
  Each block starts with:
    #Parameters = {rotation=X; ...}
    #"Frequency / THz"  "SZmax(m),Zmax(n) (rotation=X) [Re]" ...
    #---...
    data rows

Author: Anuvansh Gera
"""

from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"      # put CST exports here
OUT_DIR_DEFAULT = HERE / "outputs"
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# =============================================================================
# CONFIG
# =============================================================================

BASE = str(DATA_DIR)

CIRC_FILE = os.path.join(BASE, "S_params_sweep.txt")
LIN_FILE  = os.path.join(BASE, "S_params_sweep_linear.txt")

OUT_DIR   = BASE
FREQ_UNIT = "THz"

# S-param name → internal key (same header substrings work for both
# circular and linear port exports since CST uses the same naming)
CIRC_KEYS = {
    "SZmax(1),Zmax(1)": "r_pp",
    "SZmax(2),Zmax(2)": "r_mm",
    "SZmax(1),Zmax(2)": "r_pm",
    "SZmax(2),Zmax(1)": "r_mp",
}

LIN_KEYS = {
    "SZmax(1),Zmax(1)": "r_xx",
    "SZmax(2),Zmax(1)": "r_yx",
    "SZmax(1),Zmax(2)": "r_xy",
    "SZmax(2),Zmax(2)": "r_yy",
}

# =============================================================================
# END CONFIG
# =============================================================================


# ── parser ────────────────────────────────────────────────────────────────────

def _extract_rotation(param_line):
    """Extract rotation value from #Parameters = {rotation=X; ...}"""
    m = re.search(r'rotation=([+-]?\d+\.?\d*)', param_line)
    return float(m.group(1)) if m else None


def _identify_sparam(header_line, key_map):
    """Return internal key for a column-header comment line."""
    for cst_name, key in key_map.items():
        if cst_name in header_line:
            return key
    return None


def load_sweep_file(path, key_map):
    """
    Parse a CST parametric sweep export.

    Returns dict: {rotation_value: {param_key: (freq, s_complex)}}
    e.g. {-30.0: {"r_pp": (freq, s), ...}, 0.0: {...}, ...}
    """
    if not os.path.isfile(path):
        print(f"  ERROR — file not found: {path}")
        return {}

    # Accumulate: runs[rotation][param_key] = list of [freq, re, im]
    runs     = {}
    cur_rot  = None
    cur_key  = None
    cur_rows = []

    def _flush():
        if cur_rot is not None and cur_key is not None and cur_rows:
            if cur_rot not in runs:
                runs[cur_rot] = {}
            runs[cur_rot][cur_key] = _to_arrays(cur_rows)

    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#Parameters"):
                _flush()
                cur_rot  = _extract_rotation(line)
                cur_key  = None
                cur_rows = []

            elif line.startswith("#\"Frequency"):
                _flush()
                cur_key  = _identify_sparam(line, key_map)
                cur_rows = []

            elif line.startswith("#"):
                pass  # separator line, skip

            else:
                if cur_rot is not None and cur_key is not None:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            cur_rows.append([float(p) for p in parts[:3]])
                        except ValueError:
                            pass

    _flush()

    # Report
    rotations = sorted(runs.keys())
    print(f"  Loaded {len(rotations)} rotation values from: "
          f"{os.path.basename(path)}")
    for rot in rotations:
        found = list(runs[rot].keys())
        print(f"    rotation={rot:+.0f}°  params: {found}")

    return runs


def _to_arrays(rows):
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1] + 1j * arr[:, 2]


# ── metrics ───────────────────────────────────────────────────────────────────

FREQ_MIN = 47.0   # THz — lower bound of resonance band for peak extraction
FREQ_MAX = 58.0   # THz — upper bound of resonance band for peak extraction
ABS_GATE = 0.05   # minimum total absorption to report a valid CD value


def circ_metrics(run):
    """run = {param_key: (freq, s)}. Returns freq, A_LCP, A_RCP, CD, g, dAT."""
    freq, r_pp = run["r_pp"]
    _,    r_mm = run["r_mm"]
    _,    r_pm = run["r_pm"]
    _,    r_mp = run["r_mp"]

    Rpp = np.abs(r_pp)**2
    Rmm = np.abs(r_mm)**2
    Rpm = np.abs(r_pm)**2
    Rmp = np.abs(r_mp)**2

    A_LCP = np.clip(1 - Rpp - Rmp, 0, 1)
    A_RCP = np.clip(1 - Rmm - Rpm, 0, 1)

    # Gate: only compute CD where total absorption is meaningful
    # Outside the absorption band the denominator is near zero -> spurious spikes
    total_abs = A_LCP + A_RCP
    denom = np.where(total_abs < ABS_GATE, np.nan, total_abs)
    CD    = np.where(total_abs < ABS_GATE, 0.0, (A_LCP - A_RCP) / denom)
    g     = 2 * CD
    dAT   = Rmp - Rpm

    return freq, A_LCP, A_RCP, CD, g, dAT


def lin_metrics(run):
    """run = {param_key: (freq, s)}. Returns freq, A_x, A_y, lin_diff."""
    freq, r_xx = run["r_xx"]
    _,    r_yx = run["r_yx"]
    _,    r_xy = run["r_xy"]
    _,    r_yy = run["r_yy"]

    A_x = np.clip(1 - np.abs(r_xx)**2 - np.abs(r_yx)**2, 0, 1)
    A_y = np.clip(1 - np.abs(r_yy)**2 - np.abs(r_xy)**2, 0, 1)

    return freq, A_x, A_y, A_x - A_y


def _colormap(n):
    cmap = get_cmap("viridis")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, name + ".png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  saved -> {p}")
    plt.close(fig)


# ── Plot 1: CD spectra overlay ────────────────────────────────────────────────

def plot_cd_overlay(circ_runs):
    print("\n-- Plot 1: CD_abs spectra overlay --")
    rotations = sorted(circ_runs.keys())
    colors    = _colormap(len(rotations))

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    for rot, color in zip(rotations, colors):
        run = circ_runs[rot]
        if not all(k in run for k in ["r_pp","r_mm","r_pm","r_mp"]):
            continue
        freq, A_LCP, A_RCP, CD, g, dAT = circ_metrics(run)
        lbl = f"{rot:+.0f}°"
        axes[0].plot(freq, A_LCP, color=color, linewidth=1.2, label=lbl)
        axes[0].plot(freq, A_RCP, color=color, linewidth=1.2, linestyle="--")
        axes[1].plot(freq, CD,   color=color, linewidth=1.2, label=lbl)
        axes[2].plot(freq, dAT,  color=color, linewidth=1.2, label=lbl)

    axes[0].set_ylabel("Absorption")
    axes[0].set_title("A_LCP (solid) and A_RCP (dashed) — rotation sweep")
    axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=3)

    axes[1].axhline(0, color="k", linewidth=0.7, linestyle=":")
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].set_ylabel("CD_abs")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7, ncol=3)

    axes[2].axhline(0, color="k", linewidth=0.7, linestyle=":")
    axes[2].set_ylabel("ΔAT")
    axes[2].set_xlabel(f"Frequency ({FREQ_UNIT})")
    axes[2].grid(alpha=0.3); axes[2].legend(fontsize=7, ncol=3)

    fig.tight_layout()
    _save(fig, "sweep_cd_overlay")


# ── Plot 2: Trade-off curve ───────────────────────────────────────────────────

def plot_tradeoff(circ_runs, lin_runs):
    print("\n-- Plot 2: Trade-off curve --")

    rotations = sorted(set(circ_runs.keys()) & set(lin_runs.keys()))
    peak_CD, peak_Ax, peak_Ay, peak_dAT, rots = [], [], [], [], []

    for rot in rotations:
        cr = circ_runs[rot]
        lr = lin_runs[rot]
        if not (all(k in cr for k in ["r_pp","r_mm","r_pm","r_mp"]) and
                all(k in lr for k in ["r_xx","r_yx","r_xy","r_yy"])):
            continue
        freq_c, A_LCP, A_RCP, CD, _, dAT = circ_metrics(cr)
        freq_l, A_x, A_y, _              = lin_metrics(lr)

        # Restrict to resonance band only
        band_c = (freq_c >= FREQ_MIN) & (freq_c <= FREQ_MAX)
        band_l = (freq_l >= FREQ_MIN) & (freq_l <= FREQ_MAX)

        peak_CD.append(np.max(np.abs(CD[band_c])))
        peak_Ax.append(np.max(A_x[band_l]))
        peak_Ay.append(np.max(A_y[band_l]))
        peak_dAT.append(np.max(np.abs(dAT[band_c])))
        rots.append(rot)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(peak_CD, peak_Ax, "o-", color="royalblue",
                 label="Peak A_x", markersize=6)
    axes[0].plot(peak_CD, peak_Ay, "s--", color="tomato",
                 label="Peak A_y", markersize=6)
    for i, rot in enumerate(rots):
        axes[0].annotate(f"{rot:+.0f}°", (peak_CD[i], peak_Ax[i]),
                         fontsize=7, textcoords="offset points", xytext=(4, 3))
    axes[0].set_xlabel("Peak |CD_abs|")
    axes[0].set_ylabel("Peak Absorption")
    axes[0].set_title("Absorption vs CD Trade-off")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(peak_CD, peak_dAT, "D-", color="darkorange", markersize=6)
    for i, rot in enumerate(rots):
        axes[1].annotate(f"{rot:+.0f}°", (peak_CD[i], peak_dAT[i]),
                         fontsize=7, textcoords="offset points", xytext=(4, 3))
    axes[1].set_xlabel("Peak |CD_abs|")
    axes[1].set_ylabel("Peak |ΔAT|")
    axes[1].set_title("ΔAT vs CD Trade-off")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "tradeoff_cd_absorption")

    print(f"\n  {'Rotation':>10}  {'CD_abs':>8}  {'A_x':>8}"
          f"  {'A_y':>8}  {'ΔAT':>8}")
    for i, rot in enumerate(rots):
        print(f"  {rot:>+10.0f}°  {peak_CD[i]:>8.4f}  {peak_Ax[i]:>8.4f}"
              f"  {peak_Ay[i]:>8.4f}  {peak_dAT[i]:>8.4f}")


# ── Plot 3: Linear absorption overlay ────────────────────────────────────────

def plot_lin_overlay(lin_runs):
    print("\n-- Plot 3: Linear absorption overlay --")
    rotations = sorted(lin_runs.keys())
    colors    = _colormap(len(rotations))

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for rot, color in zip(rotations, colors):
        run = lin_runs[rot]
        if not all(k in run for k in ["r_xx","r_yx","r_xy","r_yy"]):
            continue
        freq, A_x, A_y, lin_diff = lin_metrics(run)
        lbl = f"{rot:+.0f}°"
        axes[0].plot(freq, A_x, color=color, linewidth=1.2, label=lbl)
        axes[0].plot(freq, A_y, color=color, linewidth=1.2, linestyle="--")
        axes[1].plot(freq, lin_diff, color=color, linewidth=1.2, label=lbl)

    axes[0].set_ylabel("Absorption")
    axes[0].set_title("A_x (solid) A_y (dashed) — rotation sweep")
    axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=3)

    axes[1].axhline(0, color="k", linewidth=0.7, linestyle=":")
    axes[1].set_ylabel("A_x − A_y  (linear dichroism)")
    axes[1].set_xlabel(f"Frequency ({FREQ_UNIT})")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7, ncol=3)

    fig.tight_layout()
    _save(fig, "sweep_linear_overlay")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Chirality Rotation Sweep Analysis")
    print("=" * 60)

    circ_runs = {}
    lin_runs  = {}

    if os.path.isfile(CIRC_FILE):
        print(f"\n-- Loading circular sweep: {os.path.basename(CIRC_FILE)} --")
        circ_runs = load_sweep_file(CIRC_FILE, CIRC_KEYS)
    else:
        print(f"\n  CIRC_FILE not found, skipping circular analysis.")

    if os.path.isfile(LIN_FILE):
        print(f"\n-- Loading linear sweep: {os.path.basename(LIN_FILE)} --")
        lin_runs = load_sweep_file(LIN_FILE, LIN_KEYS)
    else:
        print(f"\n  LIN_FILE not found, skipping linear analysis.")

    if circ_runs:
        plot_cd_overlay(circ_runs)

    if circ_runs and lin_runs:
        plot_tradeoff(circ_runs, lin_runs)

    if lin_runs:
        plot_lin_overlay(lin_runs)

    print("\n-- Done --")