# CosChaos — cosine-coordinate fractional Wiener chaos solver

**Software and benchmark release [v1.3.0](https://github.com/EslamAhmedGaber/fractional-wiener-chaos-solver/releases/tag/v1.3.0)**

CosChaos approximates the solution of a fractional stochastic differential equation by a polynomial in independent Gaussian coordinates. A closed-form cosine Gram matrix represents the fractional forcing, and a deterministic Galerkin system gives the polynomial coefficients. These coefficients give means and covariances directly. Evaluating the polynomial with Gaussian inputs gives samples for other observables, including the terminal ordering of two competing populations.

## Reproduce the 20 November 2026 manuscript benchmarks

The [benchmark directory](reproduction/2026-11-20) contains the version of the solver used for the three manuscript benchmarks, all simulation drivers, fixed seeds, synthetic numerical results and figure/table generators. From the repository root, run

~~~sh
python -m pip install -r reproduction/2026-11-20/requirements.txt
python reproduction/2026-11-20/scripts/reproduce.py
~~~

This regenerates the numerical tables and figures from stored records. To repeat the nine independent references, run the reference driver. To repeat the four end-to-end comparisons, run the timed verification driver after refreshing the references.

~~~sh
python reproduction/2026-11-20/scripts/refresh_references.py
python reproduction/2026-11-20/scripts/verify_timed_outputs.py
~~~

| Model | Outputs in the benchmark |
|---|---|
| Fractional Malthusian growth | Exact-law checks, mode and chaos truncation, FWHE and discrete Wick comparisons, multi-Hurst reuse |
| Logistic population growth | Mean, variance and terminal quantiles, independent WIS path references at six parameter settings |
| Coupled two-species competition | Means, the full covariance matrix and the terminal probability of reversed population ordering |

In the coupled model at H = 0.95, the recorded end-to-end times for eight-time means and covariances were 0.302 s for CosChaos and 9.79 s for control-variate WIS-RK4. Both met the stated 1% criterion. The terminal-ordering experiment used a separate degree-two polynomial and yielded 24.48%, compared with 24.79% in the independent reference. Timings are machine-dependent. See the [measured timings](reproduction/2026-11-20/evidence/end_to_end.json) and [benchmark instructions](reproduction/2026-11-20/README.md).

## Earlier experiments and citation

The [12 November reproduction package](reproduction/2026-11-12) and legacy solver and drivers remain available for the earlier experiments. The code used for the November 20 comparisons is in [the dated solver directory](reproduction/2026-11-20/evidence/solver).

Use [CITATION.cff](CITATION.cff) to cite this release. Its downloadable benchmark asset contains only the code and scientific data, together with requirements and reproduction instructions. The published file hashes are also listed in [the snapshot manifest](reproduction/2026-11-20/SHA256SUMS.txt). See [data provenance](docs/DATA_PROVENANCE.md) for the earlier datasets.
