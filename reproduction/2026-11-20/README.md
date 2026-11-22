# CosChaos benchmark code and data

This archive accompanies *CosChaos: a cosine Wiener chaos solver for stochastic differential equations driven by fractional Brownian motion*. All benchmark inputs are synthetic.

Install the packages in requirements.txt. The evidence directory contains the solver, model drivers, parameter settings, seeded reference calculations, terminal samples, recorded runtimes and output summaries. The scripts directory contains the table and figure generators, selection sensitivity, reference refresh, paired grid check, timed moment check and the reversal-probability check (check_reversal.py, which resamples the stored Problem 3 coefficients at degree two for J = 17, 33 and 65 and at degree three, and writes evidence/P3_lotka_volterra/reversal_checks.json).

Run python scripts/reproduce.py to regenerate numerical tables and vector figures from the stored results. To repeat the stochastic calculations, run python scripts/refresh_references.py and python scripts/verify_timed_outputs.py. The full problem drivers in evidence rerun the parameter grids. Timings depend on the machine.
