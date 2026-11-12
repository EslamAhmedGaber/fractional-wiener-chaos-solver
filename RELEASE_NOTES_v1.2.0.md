# CosChaos v1.2.0 — manuscript reproducibility update

This release accompanies the 12 November 2026 CosChaos manuscript.

- Adds the current five-figure and table generators and the numerical records they consume.
- Adds reproducible drivers for noisy oscillator Hurst inference, cubic rare-event sampling,
  independent replication, paired time-step refinement and selective tanh projection.
- Includes frozen experiment-engine snapshots to preserve the calculation environment.
- Retains the existing core solver and previously released experiments.
- Removes editorial wording from code comments/docstrings and replaces two machine-specific paths
  with repository-relative paths. Scientific caveats and attribution remain.
- Updates the README, citation metadata, validation report and file manifest.

The current five figures passed 40 + 8 data assertions; the five regenerated table files match their
reference text after newline normalisation. See the validation report for all checks and their scope.
The expensive reference simulations were not repeated for this publication; their records, seeds and
drivers are supplied. This release contains software/data, not the manuscript or journal documents.

No solver-wide speedup, AI training result or archival DOI is implied.
