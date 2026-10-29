"""
Multipole decomposition — Gold MIM absorber
3 frequencies: 49 THz (UP), 51 THz (resonance), 54 THz (LP)

Geometry (z axis, origin at top of antenna):
  z = 0  to +100 nm : gold ellipse antenna   ← integrate here
  z = 0  to -100 nm : PMMA spacer
  z = -100 to -250 nm: gold mirror

Files: 9-column CST export, 50 nm uniform grid
  x[nm]  y[nm]  z[nm]  JxRe  JxIm  JyRe  JyIm  JzRe  JzIm  (A/m²)
"""

from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"      # put CST exports here
OUT_DIR_DEFAULT = HERE / "outputs"
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── FILE PATHS ────────────────────────────────────────────────────────────────
FILES = {
    49.00: str(DATA_DIR / "Jx49.txt"),
    51.51: str(DATA_DIR / "Jx51.txt"),
    54.00: str(DATA_DIR / "Jx54.txt"),
}
OUTPUT = str(OUT_DIR_DEFAULT / "fig_multipole_3freq.png")

# ── PLOT STYLE ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11, 'axes.linewidth': 1.2,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 6, 'ytick.major.size': 6,
    'xtick.minor.size': 3, 'ytick.minor.size': 3,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
})


def load_J(path):
    """Load 9-column CST J export. Returns x,y,z (nm) and Jx,Jy,Jz (complex A/m²)."""
    rows = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
            parts = line.split()
            if len(parts) != 9:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    d = np.array(rows)
    x  = d[:, 0]; y = d[:, 1]; z = d[:, 2]
    Jx = d[:, 3] + 1j * d[:, 4]
    Jy = d[:, 5] + 1j * d[:, 6]
    Jz = d[:, 7] + 1j * d[:, 8]
    return x, y, z, Jx, Jy, Jz


def multipole(x_nm, y_nm, z_nm, Jx, Jy, Jz, freq_THz):
    """
    Cartesian multipole decomposition. Integrates over antenna volume only
    (z = 0 to +100 nm). Uniform 50 nm grid → dV = (50 nm)³.

    Returns array [I_ED, I_MD, I_TD, I_EQ, I_MQ] in Watts.
    """
    # ── antenna volume mask ───────────────────────────────────────────────
    ant = (z_nm >= 0) & (z_nm <= 100)
    x  = x_nm[ant] * 1e-9
    y  = y_nm[ant] * 1e-9
    z  = z_nm[ant] * 1e-9
    Jx = Jx[ant]; Jy = Jy[ant]; Jz = Jz[ant]

    dV     = (50e-9) ** 3          # m³ per grid point (uniform 50 nm mesh)
    eps0   = 8.854e-12             # F/m
    c      = 3e8                   # m/s
    omega  = 2 * np.pi * freq_THz * 1e12
    iomega = 1j * omega

    # ── Electric dipole ───────────────────────────────────────────────────
    Px = np.sum(Jx * dV) / iomega
    Py = np.sum(Jy * dV) / iomega
    Pz = np.sum(Jz * dV) / iomega
    P_sq = abs(Px)**2 + abs(Py)**2 + abs(Pz)**2

    # ── Magnetic dipole ───────────────────────────────────────────────────
    rxJx = y * Jz - z * Jy
    rxJy = z * Jx - x * Jz
    rxJz = x * Jy - y * Jx
    Mx = np.sum(rxJx * dV) / (2 * c)
    My = np.sum(rxJy * dV) / (2 * c)
    Mz = np.sum(rxJz * dV) / (2 * c)
    M_sq = abs(Mx)**2 + abs(My)**2 + abs(Mz)**2

    # ── Toroidal dipole ───────────────────────────────────────────────────
    rdJ = x * Jx + y * Jy + z * Jz
    r2  = x**2 + y**2 + z**2
    Tx = np.sum((rdJ * x - 2 * r2 * Jx) * dV) / (10 * c)
    Ty = np.sum((rdJ * y - 2 * r2 * Jy) * dV) / (10 * c)
    Tz = np.sum((rdJ * z - 2 * r2 * Jz) * dV) / (10 * c)
    T_sq = abs(Tx)**2 + abs(Ty)**2 + abs(Tz)**2

    # ── Electric quadrupole (traceless Cartesian) ─────────────────────────
    coords = [x, y, z]
    Js     = [Jx, Jy, Jz]
    Qe_mat = np.zeros((3, 3), dtype=complex)
    for a in range(3):
        for b in range(3):
            diag = (2/3) * rdJ if a == b else 0
            Qe_mat[a, b] = np.sum(
                (coords[a] * Js[b] + coords[b] * Js[a] - diag) * dV
            ) / (2 * iomega)
    Qe_sq = np.sum(abs(Qe_mat)**2)

    # ── Magnetic quadrupole ───────────────────────────────────────────────
    rxJ = np.array([rxJx, rxJy, rxJz])
    Qm_mat = np.zeros((3, 3), dtype=complex)
    for a in range(3):
        for b in range(3):
            Qm_mat[a, b] = np.sum(
                (rxJ[a] * coords[b] + rxJ[b] * coords[a]) * dV
            ) / (3 * c)
    Qm_sq = np.sum(abs(Qm_mat)**2)

    # ── Scattered power (SI, Watts) ───────────────────────────────────────
    # Jackson §9.2 / Savinov PRB 2014
    I_P  = omega**4 * P_sq  / (3 * eps0 * c**3)
    I_M  = omega**4 * M_sq  / (3 * c**3)
    I_T  = omega**6 * T_sq  / (3 * eps0 * c**5)
    I_Qe = omega**6 * Qe_sq / (5 * eps0 * c**5)
    I_Qm = omega**6 * Qm_sq / (40 * c**5)

    return np.array([I_P, I_M, I_T, I_Qe, I_Qm])


# ── COMPUTE ───────────────────────────────────────────────────────────────────
freqs_THz  = sorted(FILES.keys())
c          = 3e8
wl_um      = np.array([c / (f * 1e12) * 1e6 for f in freqs_THz])
labels     = ['ED', 'MD', 'TD', 'EQ', 'MQ']
bar_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

all_powers = {}
print(f"\n{'Freq (THz)':<12} {'λ (µm)':<8} {'ED%':>6} {'MD%':>6} {'TD%':>6} {'EQ%':>6} {'MQ%':>6}")
print("-" * 55)
for f in freqs_THz:
    x, y, z, Jx, Jy, Jz = load_J(FILES[f])
    pw = multipole(x, y, z, Jx, Jy, Jz, f)
    all_powers[f] = pw
    total = pw.sum()
    pcts  = 100 * pw / total
    wl    = c / (f * 1e12) * 1e6
    print(f"{f:<12.2f} {wl:<8.3f} " + " ".join(f"{p:>6.1f}" for p in pcts))


# ── FIGURE: grouped bar + fractional line ────────────────────────────────────
fig = plt.figure(figsize=(13, 6))
gs  = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.38)

# ── LEFT: stacked bar ─────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
n_f = len(freqs_THz)
bar_w = 0.22
x_pos = np.arange(len(labels))
freq_colors = ['#2166ac', '#4dac26', '#d01c8b']
freq_labels = [f'f={f:.0f} THz  (λ≈{c/(f*1e12)*1e6:.2f} µm)' for f in freqs_THz]
branch_tags = ['≈ UP', '≈ resonance', '≈ LP']

for i, (f, fc, ftag) in enumerate(zip(freqs_THz, freq_colors, branch_tags)):
    pw    = all_powers[f]
    total = pw.sum()
    pcts  = 100 * pw / total
    offset = (i - 1) * bar_w
    bars = ax1.bar(x_pos + offset, pw, bar_w,
                   color=fc, alpha=0.85, edgecolor='black', lw=0.8,
                   label=f'{ftag}  {freq_labels[i]}', zorder=3)
    for bar, pct in zip(bars, pcts):
        if pct > 1.0:
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() * 1.6,
                     f'{pct:.0f}%', ha='center', va='bottom',
                     fontsize=7.5, fontweight='bold', color=fc)

ax1.set_yscale('log')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, fontsize=13, fontweight='bold')
ax1.set_ylabel('Scattering power (W)', fontsize=12, fontweight='bold')
ax1.set_title('Multipole decomposition — antenna volume only\n'
              '(z = 0 → +100 nm, gold ellipses)',
              fontsize=10.5, fontweight='bold')
ax1.legend(fontsize=8.5, framealpha=0.92, edgecolor='black',
           fancybox=False, loc='lower right')
ax1.yaxis.set_minor_locator(ticker.LogLocator(subs='all', numticks=10))
ax1.grid(True, axis='y', ls='--', alpha=0.25, lw=0.8, zorder=0)
for sp in ax1.spines.values():
    sp.set_linewidth(1.2); sp.set_visible(True)
ax1.tick_params(which='both', direction='in', top=True, right=True)

# ── RIGHT: % contribution vs wavelength ───────────────────────────────────
ax2 = fig.add_subplot(gs[1])
line_styles = ['-', '-', '--', '-.', ':']

for lbl, col, ls in zip(labels, bar_colors, line_styles):
    idx  = labels.index(lbl)
    pcts = np.array([100 * all_powers[f][idx] / all_powers[f].sum()
                     for f in freqs_THz])
    if pcts.max() < 0.1:
        continue
    ax2.plot(wl_um, pcts, marker='o', ls=ls, color=col, lw=2.2,
             ms=8, markeredgecolor='white', markeredgewidth=1.0,
             label=lbl, zorder=4)

ax2.axvline(5.820, color='#8B0000', ls='--', lw=1.2, alpha=0.7)
ax2.text(5.835, 2, 'PMMA\n5.82 µm', fontsize=8, color='#8B0000', va='bottom')

# shade LP / UP regions
ax2.axvspan(wl_um.min() - 0.05, 5.75, alpha=0.07, color='#d01c8b')
ax2.axvspan(5.90, wl_um.max() + 0.05, alpha=0.07, color='#2166ac')
ax2.text(5.58, 96, 'LP', fontsize=10, color='#d01c8b',
         fontweight='bold', ha='center')
ax2.text(6.08, 96, 'UP', fontsize=10, color='#2166ac',
         fontweight='bold', ha='center')

ax2.set_xlim(wl_um.min() - 0.1, wl_um.max() + 0.1)
ax2.set_ylim(0, 100)
ax2.set_xlabel('Wavelength (µm)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Multipole contribution (%)', fontsize=12, fontweight='bold')
ax2.set_title('Multipole character\nvs polariton branch', fontsize=10.5, fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.92, edgecolor='black', fancybox=False)
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(5))
ax2.grid(True, ls='--', alpha=0.2, lw=0.7)
for sp in ax2.spines.values():
    sp.set_linewidth(1.2); sp.set_visible(True)
ax2.tick_params(which='both', direction='in', top=False, right=True)

# secondary x: wavenumber
ax2t = ax2.twiny()
ax2t.set_xlim(ax2.get_xlim())
wn_ticks = [1600, 1700, 1800]
ax2t.set_xticks([1e4 / w for w in wn_ticks])
ax2t.set_xticklabels([str(w) for w in wn_ticks], fontsize=9)
ax2t.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10, fontweight='bold')
ax2t.tick_params(which='both', direction='in', top=True)
ax2t.xaxis.set_minor_locator(ticker.AutoMinorLocator())
for sp in ax2t.spines.values():
    sp.set_linewidth(1.2)

plt.suptitle('Gold MIM absorber — Multipole decomposition (antenna volume, z = 0 → +100 nm)',
             fontsize=11.5, fontweight='bold', y=1.02)
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
plt.savefig(OUTPUT, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved → {OUTPUT}")