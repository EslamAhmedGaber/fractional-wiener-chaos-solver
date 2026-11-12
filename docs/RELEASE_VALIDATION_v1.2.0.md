# Release validation — v1.2.0

Validation date: 12 November 2026. Source: the numerical package accompanying the final CosChaos
manuscript dated 12 November 2026. This is a software/data release; manuscript PDFs and journal
correspondence are not published here.

## Checks performed

- Regenerated Figures 2–5: 40 data assertions passed.
- Regenerated the construction/output figure: 8 data assertions passed.
- Regenerated five LaTeX tables: text matches the supplied tables after newline normalisation.
- Recomputed all 18 pooled importance-sampling estimates from the original and independent replication
  moments: exact agreement with the supplied numerical records.
- Core engine: Brownian Gram identity, fractional Gram positivity/parity and a stored nonlinear cubic
  mean/variance regression passed.
- Selective backend: agreement with the full tensor solver at degrees 3 and 5, and conditional projection
  against four-dimensional Gauss–Hermite quadrature passed. Maximum error was below 1.4e-15.
- Portable Hurst driver: oscillator mean agreed with the analytic mean to 6.92e-14; the response and
  independent reference-covariance routines executed successfully.
- Portable rare-event driver: a degree-three cubic solve and discrete-coordinate whitening passed.
- All Python files parsed. The public Python sources were scanned for machine-specific drive paths,
  assistant/team references and unfinished editorial markers; none remained in the scanned categories.
- Comment/docstring cleanup was checked by comparing executable ASTs with docstrings removed.
  Separate portability changes replace two absolute paths and relocate replication outputs. Numerical
  formulas, seeds and stopping criteria were preserved.
- Tested an archive extracted into a fresh directory. This exposed a missing output-directory creation
  in the table generator; it was fixed. The fresh extraction passed manifest verification, exact table
  and pooled-estimator reproduction, the core regression and the three selective-backend checks.
  The figure generators also completed from the extracted package with all 48 assertions passing.

The root `solver/` files are byte-identical to the preceding repository version. Historical experiment
engines remain available. Newly copied text files use LF endings so the manifest survives a Git clone
on Windows. Stored numerical evidence is unchanged apart from text newline normalisation.

## Scope

The 600-data-set inference run and the multi-million-path Monte Carlo references were not rerun for this
publication. Their supplied records and experiment scripts are included. Validation distinguishes
reproducing recorded arithmetic/plots from repeating the full stochastic experiments. No new theorem,
universal performance guarantee or journal-readiness verdict is implied.

Private future-work material, unused external-competitor records and exploratory control-variate records
were not added to the current reproduction package. Existing public historical evidence remains available.
Scientific caveats, source attribution and author declarations have not been removed.

The manifest covers the released repository files except itself. Large regenerated sample arrays,
generated figure files and local check outputs are excluded from version control and can be reproduced
with the documented commands.
