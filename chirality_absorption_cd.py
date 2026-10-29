"""
chirality_analysis_mim.py  (v4 — absorption CD + linear polarisation absorptions)
==================================================================================
Correct CD metric for a MIM perfect absorber (T = 0, ground plane).

CD_absorption = (A_LCP - A_RCP) / (A_LCP + A_RCP)

where, for circular-port CST export:
  A_LCP = 1 - |r_pp|^2 - |r_mp|^2   (total reflected power under LCP in)
  A_RCP = 1 - |r_mm|^2 - |r_pm|^2   (total reflected power under RCP in)

LINEAR ABSORPTIONS (T=0, derived from circular-basis r-matrix):
  The Jones reflection matrix in circular basis:
      R_circ = [[r_pp, r_pm],
                [r_mp, r_mm]]

  Convert to linear basis (x/y) via:
      U = (1/sqrt(2)) * [[1,  1],
                         [i, -i]]
      R_lin = U @ R_circ @ U†

  Then:
      Ax = 1 - |R_lin[0,0]|^2 - |R_lin[1,0]|^2   (x-polarised input)
      Ay = 1 - |R_lin[1,1]|^2 - |R_lin[0,1]|^2   (y-polarised input)

Also computes:
  - g factor (dissymmetry): g = 2 * CD_abs
  - Asymmetric transmission: delta = |r_mp|^2 - |r_pm|^2

CST PORT SETUP:
  Port -> Zmax -> "Circular polarisation" ON
  Mode 1 = LCP, Mode 2 = RCP
  Export all 4 S-params stacked, Real and Imaginary.

Author: Anuvansh Gera
"""

from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"      # put CST exports here
OUT_DIR_DEFAULT = HERE / "outputs"
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================

INPUT_FILE = str(DATA_DIR / "Sparams_600.txt")
OUT_DIR    = str(OUT_DIR_DEFAULT / "chirality_plots")
FREQ_UNIT  = "THz"

# =============================================================================
# END CONFIG
# =============================================================================

_PARAM_KEYS = {
    "SZmax(1),Zmax(1)": "r_pp",   # LCP in -> LCP out
    "SZmax(2),Zmax(2)": "r_mm",   # RCP in -> RCP out
    "SZmax(1),Zmax(2)": "r_pm",   # RCP in -> LCP out
    "SZmax(2),Zmax(1)": "r_mp",   # LCP in -> RCP out
}


def _identify_param(header_line):
    for cst_name, key in _PARAM_KEYS.items():
        if cst_name in header_line:
            return key
    return None


def load_stacked_file(path):
    if not os.path.isfile(path):
        print(f"ERROR — file not found:\n  {path}")
        sys.exit(1)

    results  = {}
    cur_key  = None
    cur_rows = []

    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                key = _identify_param(line)
                if key is not None:
                    if cur_key is not None and cur_rows:
                        results[cur_key] = _rows_to_arrays(cur_rows)
                    cur_key  = key
                    cur_rows = []
            else:
                if cur_key is not None:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            cur_rows.append([float(p) for p in parts[:3]])
                        except ValueError:
                            pass

    if cur_key is not None and cur_rows:
        results[cur_key] = _rows_to_arrays(cur_rows)

    missing = [k for k in _PARAM_KEYS.values() if k not in results]
    if missing:
        print("ERROR — missing S-parameters:")
        rev = {v: k for k, v in _PARAM_KEYS.items()}
        for k in missing:
            print(f"  {k}  ({rev[k]})")
        sys.exit(1)

    print(f"  Loaded {len(results)} S-parameters from: {os.path.basename(path)}")
    for key, (freq, s) in results.items():
        print(f"    {key:6s}  {len(freq)} points  "
              f"freq [{freq[0]:.3f} ... {freq[-1]:.3f}] {FREQ_UNIT}")
    return results


def _rows_to_arrays(rows):
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1] + 1j * arr[:, 2]


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, name + ".png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  saved -> {p}")
    plt.close(fig)


def compute_linear_absorptions(r_pp, r_mm, r_pm, r_mp):
    """
    Convert circular-basis reflection matrix to linear (x/y) basis and
    compute Ax, Ay (absorption under x- and y-polarised incidence).

    Basis convention (CST circular ports):
      Mode 1 = LCP (+), Mode 2 = RCP (-)

    Unitary transform  U : linear -> circular
      U = (1/sqrt(2)) * [[1,  1 ],
                         [i, -i ]]

    R_lin = U^† @ R_circ @ U

    For T=0 (MIM ground plane), total absorption under each linear pol:
      Ax = 1 - |R_lin_xx|^2 - |R_lin_yx|^2
      Ay = 1 - |R_lin_yy|^2 - |R_lin_xy|^2
    """
    # Build per-frequency 2x2 circular reflection matrix
    # R_circ = [[r_pp, r_pm],
    #           [r_mp, r_mm]]
    # Shape: (N, 2, 2)
    N = len(r_pp)
    R_circ = np.stack([
        np.stack([r_pp, r_pm], axis=-1),
        np.stack([r_mp, r_mm], axis=-1),
    ], axis=-2)  # (N, 2, 2)

    # Unitary matrix  U  (columns = circular modes expressed in x/y basis)
    sq2 = np.sqrt(2)
    U = np.array([[1,  1],
                  [1j, -1j]]) / sq2  # shape (2, 2)

    # R_lin = U^H @ R_circ @ U  (broadcast over N)
    Uh = U.conj().T            # (2, 2)
    R_lin = Uh @ R_circ @ U   # (N, 2, 2)  — numpy broadcasts correctly

    # Column 0 = x-pol input, column 1 = y-pol input
    R_lin_xx = R_lin[:, 0, 0]
    R_lin_yx = R_lin[:, 1, 0]
    R_lin_xy = R_lin[:, 0, 1]
    R_lin_yy = R_lin[:, 1, 1]

    Ax = 1 - np.abs(R_lin_xx)**2 - np.abs(R_lin_yx)**2
    Ay = 1 - np.abs(R_lin_yy)**2 - np.abs(R_lin_xy)**2

    Ax = np.clip(Ax, 0, 1)
    Ay = np.clip(Ay, 0, 1)

    return Ax, Ay, R_lin_xx, R_lin_yy, R_lin_xy, R_lin_yx


def plot_absorption_cd(freq, r_pp, r_mm, r_pm, r_mp):
    print("\n-- Absorption CD (correct MIM metric) --")

    Rpp = np.abs(r_pp)**2
    Rmm = np.abs(r_mm)**2
    Rpm = np.abs(r_pm)**2
    Rmp = np.abs(r_mp)**2

    A_LCP = np.clip(1 - Rpp - Rmp, 0, 1)
    A_RCP = np.clip(1 - Rmm - Rpm, 0, 1)

    denom = np.where((A_LCP + A_RCP) < 1e-12, 1e-12, A_LCP + A_RCP)
    CD    = (A_LCP - A_RCP) / denom
    g     = 2 * CD
    delta_AT = Rmp - Rpm

    peak_CD = np.max(np.abs(CD))
    peak_g  = np.max(np.abs(g))
    peak_AT = np.max(np.abs(delta_AT))

    print(f"  Peak |CD_abs|   = {peak_CD:.4f}  "
          f"{'-> CHIRAL' if peak_CD > 0.05 else '-> weak'}")
    print(f"  Peak |g|        = {peak_g:.4f}")
    print(f"  Peak |delta_AT| = {peak_AT:.4f}")

    fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)

    axes[0].plot(freq, A_LCP, label="A_LCP", color="royalblue")
    axes[0].plot(freq, A_RCP, label="A_RCP", color="tomato", linestyle="--")
    axes[0].set_ylabel("Absorption")
    axes[0].set_title("Chiral Absorption — MIM Absorber")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(freq, CD, color="darkgreen", linewidth=1.5)
    axes[1].axhline(0,  color="k",    linewidth=0.7, linestyle=":")
    axes[1].axhline( 1, color="grey", linewidth=0.5, linestyle=":")
    axes[1].axhline(-1, color="grey", linewidth=0.5, linestyle=":")
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].set_ylabel("CD_abs")
    axes[1].grid(alpha=0.3)

    axes[2].plot(freq, g, color="purple", linewidth=1.5)
    axes[2].axhline(0, color="k", linewidth=0.7, linestyle=":")
    axes[2].set_ylim(-2.1, 2.1)
    axes[2].set_ylabel("g = 2 × CD_abs")
    axes[2].grid(alpha=0.3)

    axes[3].plot(freq, delta_AT, color="darkorange", linewidth=1.5)
    axes[3].axhline(0, color="k", linewidth=0.7, linestyle=":")
    axes[3].set_ylabel(r"$\Delta$AT = $|r_{-+}|^2 - |r_{+-}|^2$")
    axes[3].set_xlabel(f"Frequency ({FREQ_UNIT})")
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "absorption_cd_mim")
    return CD, g, delta_AT, A_LCP, A_RCP


def plot_linear_absorptions(freq, r_pp, r_mm, r_pm, r_mp):
    print("\n-- Linear Polarisation Absorptions (Ax, Ay) --")

    Ax, Ay, R_xx, R_yy, R_xy, R_yx = compute_linear_absorptions(
        r_pp, r_mm, r_pm, r_mp
    )

    print(f"  Peak Ax = {np.max(Ax):.4f}  at {freq[np.argmax(Ax)]:.3f} {FREQ_UNIT}")
    print(f"  Peak Ay = {np.max(Ay):.4f}  at {freq[np.argmax(Ay)]:.3f} {FREQ_UNIT}")

    # Linear dichroism (LD): signed difference, normalised
    LD_denom = np.where((Ax + Ay) < 1e-12, 1e-12, Ax + Ay)
    LD = (Ax - Ay) / LD_denom

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    axes[0].plot(freq, Ax, label="Ax (x-pol)", color="steelblue", linewidth=1.8)
    axes[0].plot(freq, Ay, label="Ay (y-pol)", color="firebrick",
                 linewidth=1.8, linestyle="--")
    axes[0].set_ylabel("Absorption")
    axes[0].set_title("Linear Polarisation Absorptions — MIM Absorber")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(freq, np.abs(R_xx)**2, label=r"$|r_{xx}|^2$", color="steelblue")
    axes[1].plot(freq, np.abs(R_yy)**2, label=r"$|r_{yy}|^2$", color="firebrick",
                 linestyle="--")
    axes[1].plot(freq, np.abs(R_xy)**2, label=r"$|r_{xy}|^2$", color="seagreen",
                 linestyle=":")
    axes[1].plot(freq, np.abs(R_yx)**2, label=r"$|r_{yx}|^2$", color="darkorange",
                 linestyle="-.")
    axes[1].set_ylabel("Reflectance")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(alpha=0.3)

    axes[2].plot(freq, LD, color="indigo", linewidth=1.5)
    axes[2].axhline(0, color="k", linewidth=0.7, linestyle=":")
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].set_ylabel("LD = (Ax−Ay)/(Ax+Ay)")
    axes[2].set_xlabel(f"Frequency ({FREQ_UNIT})")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "linear_absorptions_mim")
    return Ax, Ay, LD


def plot_combined_summary(freq, A_LCP, A_RCP, Ax, Ay):
    """Single overview: all four absorptions on one panel."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freq, A_LCP, label="A_LCP", color="royalblue",  linewidth=1.8)
    ax.plot(freq, A_RCP, label="A_RCP", color="tomato",     linewidth=1.8, linestyle="--")
    ax.plot(freq, Ax,    label="Ax",    color="steelblue",  linewidth=1.5, linestyle=":")
    ax.plot(freq, Ay,    label="Ay",    color="firebrick",  linewidth=1.5, linestyle="-.")
    ax.set_xlabel(f"Frequency ({FREQ_UNIT})")
    ax.set_ylabel("Absorption")
    ax.set_title("All Absorptions — MIM Absorber (T = 0)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "all_absorptions_summary")


if __name__ == "__main__":
    print("=" * 60)
    print("  Chirality Analysis v4 — MIM Absorption CD + Ax/Ay")
    print("=" * 60)

    print("\n-- Loading S-parameters --")
    data = load_stacked_file(INPUT_FILE)

    freq, r_pp = data["r_pp"]
    _,    r_mm = data["r_mm"]
    _,    r_pm = data["r_pm"]
    _,    r_mp = data["r_mp"]

    CD, g, delta_AT, A_LCP, A_RCP = plot_absorption_cd(
        freq, r_pp, r_mm, r_pm, r_mp
    )
    Ax, Ay, LD = plot_linear_absorptions(freq, r_pp, r_mm, r_pm, r_mp)
    plot_combined_summary(freq, A_LCP, A_RCP, Ax, Ay)

    print("\n-- Summary --")
    print(f"  Peak |CD_abs|   = {np.max(np.abs(CD)):.4f}")
    print(f"  Peak |g|        = {np.max(np.abs(g)):.4f}")
    print(f"  Peak |delta_AT| = {np.max(np.abs(delta_AT)):.4f}")
    print(f"  Peak Ax         = {np.max(Ax):.4f}  "
          f"at {freq[np.argmax(Ax)]:.3f} {FREQ_UNIT}")
    print(f"  Peak Ay         = {np.max(Ay):.4f}  "
          f"at {freq[np.argmax(Ay)]:.3f} {FREQ_UNIT}")
    print(f"  Peak |LD|       = {np.max(np.abs(LD)):.4f}")
    print()
    print("  What each metric means:")
    print("  CD_abs  : differential absorption LCP vs RCP (main chirality metric)")
    print("  g       : dissymmetry factor (molecular chirality sensing)")
    print("  delta_AT: asymmetric circular transmission (secondary chiral signal)")
    print("  Ax / Ay : absorption under x / y linear polarisation")
    print("  LD      : linear dichroism = (Ax-Ay)/(Ax+Ay)")