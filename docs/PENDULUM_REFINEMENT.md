# Pendulum degree refinement — v1.1.0

This experiment uses the existing general coefficient law for the additive-noise pendulum:
dQ = V dt, dV = (-sin(Q) - 0.6 V) dt + 0.4 dB_H.
H=0.9, T=10, Q(0)=1, V(0)=0. The mode count and chaos degree are refined separately.

## Fixed-mode result

J=3 is held fixed. Signed errors below are for Var Q(10) against independent full-fBm MC and independent MC driven by exactly the same finite-mode Gaussian process. These are measured deviations, not deterministic error bounds.

| Degree p | Tensor/reduced order q | Var Q(10) | Full-fBm deviation (%) | Same-mode deviation (%) | Solve time (s) |
|---|---|---|---|---|---|
| 1 | 20 | 0.186285 | -75.04 | -70.44 | 0.5 |
| 2 | 11 | 0.185367 | -75.16 | -70.59 | 0.1 |
| 4 | 65 | 0.431731 | -42.16 | -31.49 | 13.7 |
| 6 | 65 | 0.578976 | -22.43 | -8.13 | 44.7 |
| 8 | 65 | 0.641420 | -14.06 | +1.78 | 259.9 |

The degree-eight endpoint variance agrees with the same-mode MC within 0.62 sampling standard errors. Raising q from 55 to 65 changes Var Q(10) by less than 0.006% and all eight stored variances by less than 0.06%. This is an observed quadrature-refinement diagnostic.

## Retained limitations

- p=2 is slightly worse than p=1: no monotonic improvement at every degree is claimed.
- At J=3 the full-fBm velocity-variance deficit at t=10 remains 71.9%, despite agreement with the same-mode velocity reference. More noise modes are needed for this observable.
- Endpoint means differ from the same-mode MC by 2.65 and 3.72 sampling standard errors; not all moments are resolved.
- J=5, p=6 is quadrature-sensitive: increasing q=13 to q=15 changes Var Q(10) by 5.7%. The more favourable intermediate value is not presented as a validated accuracy result.
- The full-fBm and finite-mode MC references use different seeds. Their difference is not a paired estimate of mode bias.
- The p=10 exploratory run has no quadrature-convergence check and is not used as the principal result.
- Original double-well and other nonlinear limitations are unchanged.

## Reproduction

From the repository root:

    python scripts/verify_pendulum_refinement.py
    python evidence/pendulum_degree_refinement/generate_degree_report.py
    python evidence/pendulum_degree_refinement/analyze_refinement.py

The verification recomputes all moments from 31 saved coefficient arrays, reruns one small callable-drift case, verifies stored quadrature diagnostics and regenerates manuscript tables. It does not rerun all expensive campaigns.

To recompute the final degree-eight experiment (about 4.3 minutes on the release workstation):

    python evidence/pendulum_degree_refinement/run_degree_refinement.py --memory-lean --rtol 1e-9 --case 3 8 65 --output runs/pendulum_degree/final.json

To rerun the independent same-mode reference (about 97 seconds in the recorded run):

    python evidence/pendulum_degree_refinement/run_finite_mode_mc.py --steps 2000 --output runs/pendulum_degree/finite_mc.json

Use --case J P Q repeatedly for other settings. The driver refuses requests above its explicit node/coefficients budgets. Timings depend on hardware, quadrature, degree and mode count; the final J=5, p=6, q=15 run took about 11 minutes, and higher-dimensional requests can cost much more.

The memory-lean driver evaluates the same Gaussian weighted sum with one cached basis matrix instead of two; it does not reduce the number of quadrature nodes. The released core solver modules are unchanged. An identical-node comparison agrees to roundoff. The additive diffusion is entered as a constant polynomial, with an independent equivalence check.

## Files

- JSON files: all trial results, parameters, moments, timings, sampling errors and checks.
- coefficients/*.npz: all 31 numerical coefficient arrays, times and integer exponent arrays. Load with numpy.load(..., allow_pickle=False).
- final_diagnostics.json: selected results and all-time/all-state error calculations.
- comparison.csv: all completed cases, including unresolved/exploratory settings.
- Generated TeX tables/macros: the exact numerical entries used in the revised manuscript.
- The previously published v1.0.0 release remains unchanged.
