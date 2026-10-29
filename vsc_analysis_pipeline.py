"""
=============================================================================
  Plasmonic Gold MIM Absorber — Full Analysis Pipeline 
=============================================================================

"""

from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"      # put CST exports here
OUT_DIR_DEFAULT = HERE / "outputs"
BASELINE_FILE    = str(DATA_DIR / "baseline_non lorentzian.txt")
LORENTZ_FILE_3F  = str(DATA_DIR / "07to130_lorentzian.txt")
BASELINE_FILE_3F = str(DATA_DIR / "07to130non_lorentzian.txt")
LORENTZ_FILE_4C  = str(DATA_DIR / "07to130_lorentzian.txt")
BASELINE_FILE_4C = str(DATA_DIR / "07to130non_lorentzian.txt")
OUTPUT_DIR       = str(OUT_DIR_DEFAULT)


# =============================================================================
#  No edits needed below this line
# =============================================================================

import os, sys, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size':        11,
    'axes.linewidth':   1.2,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.top':        True,
    'ytick.right':      True,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width':1.2,
    'ytick.major.width':1.2,
    'xtick.minor.width':0.8,
    'ytick.minor.width':0.8,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
})


def _outpath(fn):
    return os.path.join(OUTPUT_DIR, fn)


def _apply_box(ax):
    """Enforce full box frame + mirrored inward ticks on any axis."""
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
        sp.set_visible(True)
    ax.tick_params(which='both', direction='in',
                   top=True, right=True, bottom=True, left=True)


# =============================================================================
#  DATA LOADERS
# =============================================================================

def load_single_block(path):
    data = np.loadtxt(path, comments='#', delimiter='\t')
    return data[:, 0], data[:, 1]


def parse_multi_block(fname):
    with open(fname, 'r') as f:
        content = f.read()
    blocks = content.split('#Parameters')[1:]
    results = {}
    for block in blocks:
        lines = block.strip().split('\n')
        try:
            S_val = float(re.search(r'S=([\d.]+)', lines[0]).group(1))
        except:
            continue
        if S_val in results:
            continue
        data_lines = [l for l in lines[3:] if l.strip() and not l.startswith('#')]
        wl, A = [], []
        for l in data_lines:
            parts = l.strip().split('\t')
            if len(parts) == 2:
                try:
                    wl.append(float(parts[0]))
                    A.append(float(parts[1]))
                except:
                    pass
        if len(wl) > 0:
            results[S_val] = (np.array(wl), np.array(A))
    return results


def load_J_file(path):
    data = np.loadtxt(path, skiprows=2)
    x  = data[:, 0]  # nm
    y  = data[:, 1]  # nm
    z  = data[:, 2]  # nm
    Jx = data[:, 3] + 1j * data[:, 4]
    Jy = data[:, 5] + 1j * data[:, 6]
    Jz = data[:, 7] + 1j * data[:, 8]
 
    # grid step sizes
    dx = float(np.unique(np.diff(np.unique(x)))[0])
    dy = float(np.unique(np.diff(np.unique(y)))[0])
    area = np.full(len(x), dx * dy)  # nm² — dz handled by _multipole_moments
 
    # threshold out air/vacuum voxels (J ~ numerical noise)
    Jmag = np.sqrt(np.abs(Jx)**2 + np.abs(Jy)**2 + np.abs(Jz)**2)
    mask = Jmag > Jmag.max() * 1e-4
    return x[mask], y[mask], z[mask], Jx[mask], Jy[mask], Jz[mask], area[mask]


# =============================================================================
#  STEP 1 — Half-max crossing → gamma_c
# =============================================================================

def step1_halfmax_gamma_c(wl, A, wl_min=5400, wl_max=6400):
    print("\n" + "="*60)
    print("STEP 1 — Half-max crossing")
    print("="*60)
    mask = (wl >= wl_min) & (wl <= wl_max)
    wl_w, A_w = wl[mask], A[mask]
    peak_idx = np.argmax(A_w)
    peak_wl  = wl_w[peak_idx]
    peak_A   = A_w[peak_idx]
    half_max = peak_A / 2
    print(f"Peak at {peak_wl:.2f} nm,  A_max = {peak_A:.4f}")

    left_mask = wl_w < peak_wl
    left_wl, left_A = wl_w[left_mask], A_w[left_mask]
    left_cross = np.where(left_A <= half_max)[0]
    valid_left = len(left_cross) > 0 and left_cross[-1] + 1 < len(left_A)
    if valid_left:
        i1 = left_cross[-1]
        wl_left_half = left_wl[i1] + (half_max - left_A[i1]) * \
                       (left_wl[i1+1] - left_wl[i1]) / (left_A[i1+1] - left_A[i1])
    else:
        wl_left_half = None

    right_mask = wl_w > peak_wl
    right_wl, right_A = wl_w[right_mask], A_w[right_mask]
    right_cross = np.where(right_A <= half_max)[0]
    i2 = right_cross[0]
    wl_right_half = right_wl[i2-1] + (half_max - right_A[i2-1]) * \
                    (right_wl[i2] - right_wl[i2-1]) / (right_A[i2] - right_A[i2-1])

    if valid_left:
        FWHM_nm  = wl_right_half - wl_left_half
        FWHM_meV = abs(1239.84/wl_left_half - 1239.84/wl_right_half) * 1000
    else:
        half_width_right = wl_right_half - peak_wl
        FWHM_nm  = 2 * half_width_right
        FWHM_meV = 2 * abs(1239.84/peak_wl - 1239.84/wl_right_half) * 1000

    gamma_c     = FWHM_meV / 2
    E_resonance = 1239.84 * 1000 / peak_wl
    Q           = E_resonance / FWHM_meV
    print(f"FWHM = {FWHM_meV:.2f} meV  →  gamma_c = {gamma_c:.2f} meV,  Q = {Q:.2f}")
    return dict(peak_wl=peak_wl, peak_A=peak_A,
                FWHM_nm=FWHM_nm, FWHM_meV=FWHM_meV,
                gamma_c_meV=gamma_c, Q=Q)


# =============================================================================
#  STEP 2 — Lorentzian fit → omega_c, gamma_c
# =============================================================================

def _lorentzian(wl, wl0, fwhm, amp, off):
    h = fwhm / 2.0
    return off + amp * h**2 / ((wl - wl0)**2 + h**2)


def step2_lorentzian_fit(wl, A, window=(5400, 6100)):
    print("\n" + "="*60)
    print("STEP 2 — Lorentzian fit")
    print("="*60)
    mask = (wl >= window[0]) & (wl <= window[1])
    wl_w, A_w = wl[mask], A[mask]
    peak_idx  = np.argmax(A_w)
    wl0_guess = wl_w[peak_idx]
    amp_guess = A_w[peak_idx] - np.min(A_w)
    off_guess = np.min(A_w)
    popt, _   = curve_fit(_lorentzian, wl_w, A_w,
                           p0=[wl0_guess, 300., amp_guess, off_guess],
                           bounds=([5600, 200, 0, -0.1], [6200, 500, 2, 0.5]),
                           maxfev=5000)
    wl0, fwhm_nm = popt[0], abs(popt[1])
    E_left    = 1239.84 / (wl0 - fwhm_nm/2)
    E_right   = 1239.84 / (wl0 + fwhm_nm/2)
    fwhm_meV  = abs(E_left - E_right) * 1000
    gamma_c   = fwhm_meV / 2
    E_res     = 1239.84e3 / wl0
    Q         = E_res / fwhm_meV
    print(f"λ_c = {wl0:.2f} nm,  Δλ = {fwhm_nm:.2f} nm")
    print(f"FWHM = {fwhm_meV:.2f} meV  →  gamma_c = {gamma_c:.2f} meV,  Q = {Q:.2f}")
    return dict(wl0_nm=wl0, fwhm_nm=fwhm_nm,
                fwhm_meV=fwhm_meV, gamma_c_meV=gamma_c,
                Q=Q, omega_c_meV=E_res,
                fit_wl=wl_w, fit_A=_lorentzian(wl_w, *popt))


# =============================================================================
#  STEP 3 — Anti-crossing plot 
# =============================================================================

def step3_anticrossing(lorentz_file, baseline_file,
                       gamma_c=5.48, gamma_m=1.122,
                       wl_min=4500, wl_max=8000,
                       dip_lo=5200, dip_hi=6800):
    print("\n" + "="*60)
    print("STEP 3 — Anti-crossing plot")
    print("="*60)

    lorentz  = parse_multi_block(lorentz_file)
    baseline = parse_multi_block(baseline_file)
    common   = sorted(set(lorentz.keys()) & set(baseline.keys()))

    peak_minus, peak_plus, S_good = [], [], []
    for S in common:
        wl_l, A_l = lorentz[S]
        mask = (wl_l >= wl_min) & (wl_l <= wl_max)
        wl_w, A_lw = wl_l[mask], A_l[mask]
        peaks_idx, _ = find_peaks(A_lw, prominence=0.015)
        dips_idx,  _ = find_peaks(-A_lw, prominence=0.015)
        dip_wls  = wl_w[dips_idx]
        dip_cand = dip_wls[(dip_wls > dip_lo) & (dip_wls < dip_hi)]
        if len(dip_cand) == 0:
            continue
        dip_wl   = dip_cand[0]
        peak_wls = wl_w[peaks_idx]
        lp = peak_wls[peak_wls < dip_wl]
        rp = peak_wls[peak_wls > dip_wl]
        if len(lp) == 0 or len(rp) == 0:
            continue
        peak_minus.append(lp[-1])
        peak_plus.append(rp[0])
        S_good.append(S)

    lam_m  = np.array(peak_minus)
    lam_p  = np.array(peak_plus)
    S_arr  = np.array(S_good)
    Omega_meV = np.abs(1239.84/lam_p - 1239.84/lam_m) * 1000
    min_idx   = np.argmin(Omega_meV)

    # ── convert to meV for right axis ────────────────────────────────────
    E_p_meV = 1239.84e3 / lam_p          # meV
    E_m_meV = 1239.84e3 / lam_m

    fig, ax = plt.subplots(figsize=(7.5, 5.8))


    ax.axhline(5.825, color='#2166ac', lw=2, ls='--', alpha=0.5,
               label='Bare molecule (5.82 µm)')
    ax.scatter(S_arr, lam_p/1000, color='#d6604d', s=100, zorder=4,
               edgecolors='darkred', lw=1.5, label=r'$\lambda_+$ (upper polariton)')
    ax.scatter(S_arr, lam_m/1000, color='#2166ac', s=100, zorder=4,
               edgecolors='darkblue', lw=1.5, label=r'$\lambda_-$ (lower polariton)')
    ax.plot(S_arr, lam_p/1000, color='#d6604d', lw=2.5, alpha=0.8)
    ax.plot(S_arr, lam_m/1000, color='#2166ac', lw=2.5, alpha=0.8)

    ax.axvspan(S_arr.min(), S_arr.max(), alpha=0.12, color='gold',
               label=f'Strong coupling ($\\gamma_c$={gamma_c:.2f} meV)')

    ax.set_xlabel('Scale factor  S', fontsize=13, fontweight='bold')
    ax.set_ylabel('Peak wavelength (µm)', fontsize=13, fontweight='bold')
    ylim = (4.0, 8.0)
    ax.set_ylim(*ylim)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # ── RIGHT AXIS: meV scale ─────────────────────────────────────────────
    ax2 = ax.twinx()
    # map µm → meV:  E(meV) = 1239.84e3 / (λ_µm * 1000) = 1239.84 / λ_µm
    ax2.set_ylim(1239.84 / ylim[1], 1239.84 / ylim[0])
    ax2.set_ylabel('Energy (meV)', fontsize=12, fontweight='bold')
    ax2.tick_params(which='both', direction='in', right=True)
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    for sp in ax2.spines.values():
        sp.set_linewidth(1.2)

    _apply_box(ax)
    #ax.set_title('Anti-crossing — Gold MIM absorber',
     #        fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.95,
              edgecolor='black', fancybox=True)
    ax.grid(True, ls='--', alpha=0.25, lw=0.8)

    plt.tight_layout()
    out = _outpath('fig3f_anticrossing.eps')
    plt.savefig(out, format='eps', bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")

    # print table
    print(f"\n{'S':<6} {'λ_+ (nm)':<12} {'λ_- (nm)':<12} {'Ω (meV)':<10} {'g (meV)':<10} {'g/γ_c':<8}")
    print("-"*58)
    for S, lp, lm, Om in zip(S_arr, lam_p, lam_m, Omega_meV):
        print(f"{S:<6.2f} {lp:<12.2f} {lm:<12.2f} {Om:<10.2f} {Om/2:<10.2f} {Om/2/gamma_c:<8.3f}")


# =============================================================================
#  STEP 4 — Waterfall plots 
# =============================================================================

def _wl_to_wn(wl_um):
    """µm → cm⁻¹"""
    return 1e4 / wl_um

def _make_stacked(delta_dict, S_vals, ylabel, title, out_path,
                  display_scale=0.40, offset_step=0.45,
                  wl_lo=None, wl_hi=8.0,
                  global_max_override=None):
    n      = len(S_vals)
    colors = [plt.cm.rainbow_r(i / max(n - 1, 1)) for i in range(n)]

    if wl_lo is None:
        wl_lo = min(delta_dict[S][0].min() for S in S_vals)

    fig, ax = plt.subplots(figsize=(7, 13))

# Explicit margins — leave enough room for the top wavenumber axis
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.07, top=0.93)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    if global_max_override is not None:
        global_max = global_max_override
    else:
        global_max = max(np.max(np.abs(delta_dict[S][1])) for S in S_vals)
    if global_max == 0:
        global_max = 1.0

    for i, S in enumerate(S_vals):
        wl, dy = delta_dict[S]
        offset = i * offset_step
        if abs(S - 1.0) < 1e-6:
            color = '#333333'
            lw    = 2.0
        else:
            color = colors[i]
            lw    = 1.8

        dy_sc = (dy / global_max) * display_scale

        ax.plot([wl_lo, wl_hi], [offset] * 2, color='#bbbbbb', lw=0.7, zorder=1)
        ax.plot(wl, dy_sc + offset, color=color, lw=lw, zorder=2)
        ax.fill_between(wl, offset, dy_sc + offset, color=color, alpha=0.10, zorder=1)

        ax.text(wl_lo - 0.02, offset, f'S={S:.2f}',
                fontsize=8.5, va='center', ha='right',
                color='#222222',
                fontweight='bold' if abs(S - 1.0) < 1e-6 else 'normal')

    ax.axvline(5.785, color='black', ls='--', lw=1.2, alpha=0.6, zorder=3)
    ax.text(5.785 + 0.04, n * offset_step + 0.10, '5.785 µm',
            fontsize=9, ha='left', va='bottom', fontweight='bold')

  #  bar_x = wl_lo + 0.10
   # bar_b = -0.38
   # bar_h = 0.5 * display_scale
   # ax.annotate('', xy=(bar_x, bar_b + bar_h), xytext=(bar_x, bar_b),
   #             arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
   # ax.text(bar_x + 0.06, bar_b + bar_h / 2, '50 %',
   #        fontsize=10, va='center', ha='left', fontweight='bold')

    ax.set_xlim(wl_lo - 0.35, wl_hi)
    ax.set_ylim(-0.15, n * offset_step + 0.28)
    ax.yaxis.set_tick_params(labelleft=False, labelright=False)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.set_xlabel('Wavelength (µm)', fontsize=13, fontweight='bold')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax.tick_params(which='major', direction='in',
                width=1.2, length=6, top=True, bottom=True, left=True, right=True)
    ax.tick_params(which='minor', direction='in',
                width=0.8, length=3, top=True, bottom=True, left=True, right=True)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    wn_lo  = 1e4 / wl_hi
    wn_hi  = 1e4 / wl_lo
    wn_ticks = np.arange(np.ceil(wn_lo / 250) * 250,
                         np.floor(wn_hi / 250) * 250 + 1, 250)
    ax_top.set_xticks([1e4 / w for w in wn_ticks])
    ax_top.set_xticklabels([f'{int(w)}' for w in wn_ticks], fontsize=9)
    ax_top.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=11, fontweight='bold')
    ax_top.tick_params(which='both', direction='in', top=True)
    ax_top.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    for sp in ax_top.spines.values():
        sp.set_linewidth(1.2)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.2)

 #   ax.set_title(title, fontsize=11, fontweight='bold', pad=28)
    fig.savefig(out_path, format='eps', facecolor='white')
    plt.close(fig)
    print(f"Saved → {out_path}")

def step4_A_waterfalls(lorentz_file, baseline_file,
                       wl_min=4000, wl_max=8000):
    print("\n" + "="*60)
    print("STEP 4 — Waterfall plots")
    print("="*60)
    lorentz  = parse_multi_block(lorentz_file)
    baseline = parse_multi_block(baseline_file)
    common = sorted(S for S in set(lorentz.keys()) & set(baseline.keys()) if S >= 0.80)

    A_lorentz_all  = {}
    A_baseline_all = {}
    wl_data_min    = np.inf

    for S in common:
        wl_b, A_b = baseline[S]
        wl_l, A_l = lorentz[S]
        mask = (wl_b >= wl_min) & (wl_b <= wl_max)
        wl_w = wl_b[mask]
        A_bw = A_b[mask]
        A_lw = np.interp(wl_w, wl_l, A_l)
        wl_um = wl_w / 1000
        A_lorentz_all[S]  = (wl_um, A_lw)
        A_baseline_all[S] = (wl_um, A_bw)
        wl_data_min = min(wl_data_min, wl_um.min())

    S_ordered  = sorted(A_lorentz_all.keys())
    wl_lo      = wl_data_min
    S_min      = min(S_ordered)
    S_max      = max(S_ordered)

    # shared scale so 50% bar means the same thing in both figures
    gmax_lor   = max(np.max(np.abs(A_lorentz_all[S][1]))  for S in S_ordered)
    gmax_bas   = max(np.max(np.abs(A_baseline_all[S][1])) for S in S_ordered)
    shared_max = max(gmax_lor, gmax_bas)

    _make_stacked(
        A_lorentz_all, S_ordered,
        ylabel=r'Absorption',
        title=(f'Gold MIM absorber — Absorptance with PMMA (Lorentz)\n'
               f'Scale factor sweep  S = {S_min:.2f} → {S_max:.2f}'),
        out_path=_outpath('fig4c_A_Lorentz.eps'),
        wl_lo=wl_lo, wl_hi=8.0,
        global_max_override=shared_max)

    _make_stacked(
        A_baseline_all, S_ordered,
        ylabel=r'Absorption',
        title=(f'Gold MIM absorber — Bare absorptance (no PMMA)\n'
               f'Scale factor sweep  S = {S_min:.2f} → {S_max:.2f}'),
        out_path=_outpath('fig4c_A_baseline.eps'),
        wl_lo=wl_lo, wl_hi=8.0,
        global_max_override=shared_max)


# =============================================================================
#  STEP 5 — Multipole decomposition  (Cartesian, periodic unit cell)
# =============================================================================
#
#  Formulas follow Evlyukhin et al. (ACS Photonics 2019) and
#  Savinov et al. (Phys. Rev. B 2014), adapted for a finite-thickness
#  slab scatterer in a periodic unit cell.
#
#  We integrate over the antenna volume only (z = -250 … 0 nm, non-zero J).
#  Each mesh point i contributes volume dV_i = Area_i [nm²] × dz_i [nm].
#  dz is estimated per z-layer from the layer spacing.
#
#  Multipole moments (SI):
#    P  = (1/iω) ∫ J dV                      [C·m]
#    M  = (1/2c) ∫ (r × J) dV                [A·m²]
#    T  = (1/10c) ∫ [(r·J)r − 2r²J] dV      [A·m³]   (toroidal)
#    Q_e_αβ = (1/iω2) ∫ [r_α J_β + r_β J_α
#              − (2/3)δ_αβ (r·J)] dV         [C·m²]
#    Q_m_αβ = (1/3c) ∫ [(r×J)_α r_β
#              + (r×J)_β r_α] dV             [A·m³]
#
#  Scattered power (proportional to, arbitrary units for comparison):
#    I_P   ∝ ω⁴ |P|²
#    I_M   ∝ ω⁴ |M|²
#    I_T   ∝ ω⁶ |T|²
#    I_Qe  ∝ ω⁶ |Q_e|²_F   (Frobenius norm squared of traceless tensor)
#    I_Qm  ∝ ω⁶ |Q_m|²_F
#
#  Units cancel in the ratio plots, so the absolute prefactors don't matter.
#


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == '__main__':

    # Steps 1 & 2 — bare plasmon
    wl, A       = load_single_block(BASELINE_FILE)
    r1          = step1_halfmax_gamma_c(wl, A)
    r2          = step2_lorentzian_fit(wl, A)
    gamma_c_fit = r2['gamma_c_meV']
    print(f"\n→ Using gamma_c = {gamma_c_fit:.3f} meV for Steps 3 & 4")

    # Step 3 — anti-crossing
    step3_anticrossing(LORENTZ_FILE_3F, BASELINE_FILE_3F, gamma_c=gamma_c_fit)

    # Step 4 — waterfalls
    step4_A_waterfalls(LORENTZ_FILE_4C, BASELINE_FILE_4C)


    print("\n✓  Pipeline complete.")