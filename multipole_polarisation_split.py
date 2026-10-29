"""
multipole_polarisation.py
=========================
Same as multipole_only.py but runs the decomposition three times
on the same J file — once keeping only Jx, once only Jy, once only Jz.
Prints a comparison table and saves a three-panel percentage figure.
"""

from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"      # put CST exports here
OUT_DIR_DEFAULT = HERE / "outputs"
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          C O N F I G                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

J_FILE      = str(DATA_DIR / "Jx_3200.txt")
OUTPUT_PATH = str(OUT_DIR_DEFAULT / "multipole_polarisation.png")

ANT_Z_MIN   =   0.0    # nm
ANT_Z_MAX   = 100.0    # nm
DX_NM       =  50.0
DY_NM       =  50.0
DZ_NM       =  10.0
FREQ_THz    =  51.0

E0_ACTUAL   = 1.0084e8   # V/m
UNIT_CELL_NM = 3200.0    # nm

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     C O N S T A N T S                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

EPS0    = 8.854187817e-12
C_LIGHT = 2.99792458e8
dV      = (DX_NM*1e-9) * (DY_NM*1e-9) * (DZ_NM*1e-9)
I_INC   = 0.5 * EPS0 * C_LIGHT * E0_ACTUAL**2
A_CELL  = (UNIT_CELL_NM*1e-9)**2

LABELS  = ['ED','MD','TD','EQ','MQ']
POL_LABELS = ['Full J', 'Jx only', 'Jy only', 'Jz only']

MP_STYLES = {
    'ED': dict(color='#e41a1c'),
    'MD': dict(color='#377eb8'),
    'TD': dict(color='#4daf4a'),
    'EQ': dict(color='#984ea3'),
    'MQ': dict(color='#ff7f00'),
}

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                       H E L P E R S                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

def load_J(path):
    rows = []
    with open(path,'r') as fh:
        for i,line in enumerate(fh):
            if i < 2: continue
            parts = line.split()
            if len(parts) != 9: continue
            try: rows.append([float(p) for p in parts])
            except ValueError: continue
    if not rows:
        raise RuntimeError(f"No valid data in {path}")
    d  = np.array(rows)
    return (d[:,0], d[:,1], d[:,2],
            d[:,3]+1j*d[:,4],
            d[:,5]+1j*d[:,6],
            d[:,7]+1j*d[:,8])


def compute_multipoles(x_nm, y_nm, z_nm, Jx, Jy, Jz, freq_THz):
    mask = (z_nm >= ANT_Z_MIN) & (z_nm <= ANT_Z_MAX)
    x  = x_nm[mask]*1e-9;  y = y_nm[mask]*1e-9;  z = z_nm[mask]*1e-9
    Jx = Jx[mask];  Jy = Jy[mask];  Jz = Jz[mask]

    omega  = 2*np.pi*freq_THz*1e12
    iomega = 1j*omega

    # ED
    Px = np.sum(Jx)*dV/iomega
    Py = np.sum(Jy)*dV/iomega
    Pz = np.sum(Jz)*dV/iomega
    P_sq = abs(Px)**2 + abs(Py)**2 + abs(Pz)**2

    # MD
    rxJx = y*Jz - z*Jy
    rxJy = z*Jx - x*Jz
    rxJz = x*Jy - y*Jx
    Mx = np.sum(rxJx)*dV/(2*C_LIGHT)
    My = np.sum(rxJy)*dV/(2*C_LIGHT)
    Mz = np.sum(rxJz)*dV/(2*C_LIGHT)
    M_sq = abs(Mx)**2 + abs(My)**2 + abs(Mz)**2

    # TD
    rdJ = x*Jx + y*Jy + z*Jz
    r2  = x**2 + y**2 + z**2
    Tx = np.sum(rdJ*x - 2*r2*Jx)*dV/(10*C_LIGHT)
    Ty = np.sum(rdJ*y - 2*r2*Jy)*dV/(10*C_LIGHT)
    Tz = np.sum(rdJ*z - 2*r2*Jz)*dV/(10*C_LIGHT)
    T_sq = abs(Tx)**2 + abs(Ty)**2 + abs(Tz)**2

    # EQ
    coords = [x,y,z];  Js = [Jx,Jy,Jz]
    Qe_mat = np.zeros((3,3),dtype=complex)
    for a in range(3):
        for b in range(3):
            diag = (2/3)*rdJ if a==b else 0.0
            Qe_mat[a,b] = np.sum(coords[a]*Js[b]+coords[b]*Js[a]-diag)*dV*(1j/omega)
    Qe_sq = np.sum(np.abs(Qe_mat)**2)

    # MQ
    rxJ = np.array([rxJx,rxJy,rxJz])
    Qm_mat = np.zeros((3,3),dtype=complex)
    for a in range(3):
        for b in range(3):
            Qm_mat[a,b] = np.sum(rxJ[a]*coords[b]+rxJ[b]*coords[a])*dV/(3*C_LIGHT)
    Qm_sq = np.sum(np.abs(Qm_mat)**2)

    I_ED = omega**4*P_sq  /(3*EPS0*C_LIGHT**3)
    I_MD = omega**4*M_sq  /(3*C_LIGHT**3)
    I_TD = omega**6*T_sq  /(3*EPS0*C_LIGHT**5)
    I_EQ = omega**6*Qe_sq /(5*EPS0*C_LIGHT**5)
    I_MQ = omega**6*Qm_sq /(40*C_LIGHT**5)

    return np.array([I_ED,I_MD,I_TD,I_EQ,I_MQ])


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          M A I N                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    x_nm, y_nm, z_nm, Jx, Jy, Jz = load_J(J_FILE)
    print(f"Loaded {len(x_nm)} voxels from {os.path.basename(J_FILE)}")

    # Four runs: full J, Jx-only, Jy-only, Jz-only
    cases = {
        'Full J':  (Jx,              Jy,              Jz             ),
        'Jx only': (Jx,              np.zeros_like(Jy), np.zeros_like(Jz)),
        'Jy only': (np.zeros_like(Jx), Jy,            np.zeros_like(Jz)),
        'Jz only': (np.zeros_like(Jx), np.zeros_like(Jy), Jz        ),
    }

    results = {}
    print(f"\n{'Case':<12}" + "".join(f"{lb:>8}%" for lb in LABELS)
          + "".join(f"  σ_{lb}(µm²)" for lb in LABELS))
    print("─"*120)

    for name, (jx,jy,jz) in cases.items():
        pw    = compute_multipoles(x_nm, y_nm, z_nm, jx, jy, jz, FREQ_THz)
        total = pw.sum()
        pcts  = 100*pw/total
        sigma = pw/I_INC*1e12   # µm²
        results[name] = (pw, pcts, sigma)
        print(f"{name:<12}"
              + "".join(f"{p:>8.1f}" for p in pcts)
              + "".join(f"  {s:>10.3e}" for s in sigma))

    print(f"\n  I_incident = {I_INC:.4e} W/m²")

    # ── Figure: 2×2 bar chart grid ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    colors = [MP_STYLES[lb]['color'] for lb in LABELS]

    for ax, (name, (pw, pcts, sigma)) in zip(axes, results.items()):
        bars = ax.bar(LABELS, pcts, color=colors, edgecolor='white',
                      linewidth=0.8, width=0.6)
        for bar, p in zip(bars, pcts):
            if p > 1.0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                        f'{p:.1f}%', ha='center', va='bottom', fontsize=9,
                        fontweight='bold')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Contribution (%)' if ax in [axes[0],axes[2]] else '')
        ax.set_ylim(0, 110)
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.tick_params(which='both', direction='in', right=True, top=True)
        sigma_total = pw.sum()/I_INC*1e12
        ax.text(0.97, 0.97, f'σ_total = {sigma_total:.3e} µm²',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8.5, color='#333333')
        for sp in ax.spines.values():
            sp.set_linewidth(1.2)

    fig.suptitle(
        f'Multipole decomposition by current polarisation\n'
        f'{FREQ_THz} THz  |  z = {ANT_Z_MIN:.0f}–{ANT_Z_MAX:.0f} nm',
        fontsize=13, fontweight='bold'
    )
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif']})
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved → {OUTPUT_PATH}")
    print("Done.")