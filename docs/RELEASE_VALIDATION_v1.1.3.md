# Release validation — v1.1.3

Publication checks on 6 November 2026:

- All 87 supplied staging-file SHA256 entries matched before integration.
- Current solver: Brownian Gram identity, fractional Gram positivity, and stored nonlinear cubic mean/variance regression passed.
- Degree-one coordinate shares were recomputed at H=0.55, 0.75 and 0.95; maximum absolute difference from stored shares was 6.7e-16.
- Figure 9 was regenerated in both layouts and visually inspected. Figure S8 was regenerated. Raster images were not byte/pixel identical across the rendering environments; no pixel-identity claim is made here.
- Python experiment scripts parsed successfully.
- The large Monte Carlo and anisotropic high-degree experiments were not rerun for publication. Their supplied evidence, seeds and scripts are included.

The current engine is solver/. The older engine and regression tests under evidence/solver_snapshot/ remain for historical reproducibility.
The integrated release ZIP also includes the legacy plot-coordinate inputs needed by make_figures_v5.py; these were absent from the standalone staging directory.
Original staging hashes describe that source snapshot, including its original README. MANIFEST_SHA256.json is the authoritative integrated-release hash list.
Scientific manuscript PDFs, author declarations and numerical evidence have not been edited by this publication step.

