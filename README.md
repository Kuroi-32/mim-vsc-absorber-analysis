# Mid-IR MIM absorber: vibrational strong coupling and multipole analysis

Post-processing code for a gold metal–insulator–metal absorber: a dual-ellipse V-shaped antenna on a PMMA spacer over an Au mirror.
The cavity is tuned to the PMMA C=O stretch (1718 cm⁻¹, ≈5.76 µm).
All inputs are CST Studio Suite exports.
The work was done during a research internship with Prof. Govind Dayal (University of Delhi). A manuscript is in preparation.

## Scripts

| Script | Purpose |
|---|---|
| `vsc_analysis_pipeline.py` | Absorption spectra, Lorentzian cavity fit (λ_c, γ_c, Q), anti-crossing over the antenna-scaling sweep, coupled-oscillator fit (Ω, g, g/ω_c, Ω/γ_c); journal-style dual-axis (µm / cm⁻¹) figures |
| `multipole_decomposition.py` | Cartesian multipole decomposition (ED, MD, TD, EQ, MQ) of CST volume-current exports, following Evlyukhin et al., PRB 2019 and Savinov et al., PRB 2014 |
| `multipole_polarisation_split.py` | Same decomposition run on Jx, Jy and Jz separately |
| `chirality_absorption_cd.py` | Absorption circular dichroism and linear absorptions from a circular-port CST S-parameter export (Jones-matrix basis change) |
| `chirality_rotation_sweep.py` | CD and absorption over an antenna-rotation parametric sweep |

## Usage

```bash
pip install -r requirements.txt
# put the CST exports in data/ (file names are set in each script's CONFIG block)
python multipole_decomposition.py
```

Current exports are 9-column CST ASCII: `x y z JxRe JxIm JyRe JyIm JzRe JzIm`, with a 2-line header.
Figures are written to `outputs/`.

Simulation data is not included.
Check: a synthetic uniform x-directed current in the antenna volume returns 99.9% ED, as expected.

## License
MIT
