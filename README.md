# Fractional Wiener Chaos Solver

Research code accompanying *A fractional Wiener chaos solver for stochastic differential equations driven by fractional Brownian motion: closed-form cosine Gram matrices and deterministic coefficient equations*.

**Release v1.1.0 — 28 October 2026.** This public research-software release is not a claim of journal acceptance.

Given drift, diffusion, an initial state, H, T, mode count J and chaos degree p, the solver generates and integrates deterministic Galerkin coefficient equations. Means and covariances follow from the coefficients. The user supplies no model-specific solution kernel.

The [method and six-step algorithm](docs/METHOD.md) describe the basis, coefficient law, moments and supported equation class.

## Scope

- Standard fractional Brownian motion on [0,T], 1/2 <= H < 1; Wick/Skorohod noise.
- One scalar noise driver and scalar or vector states.
- Cosine (default) and half-integer sine coordinates; closed-form fractional Gram matrix and Cholesky coordinates.
- Polynomial fields through cubic degree; degree-one Gaussian reduction for callable fields; higher-degree tensor quadrature within a node budget.
- Scalar linear Caputo-memory branch.
- H-independent correlated coefficient equations only for the supported affine class with H-independent model data. Ordinary nonlinear products generally depend on H.

This is not a universal solver for arbitrary stochastic processes. Mode and degree changes are diagnostics, not automatic error certificates for all nonlinear equations. The double-well limitation is retained. New fixed-mode pendulum experiments substantially improve the displacement variance at higher chaos degree, with remaining mode, mean and higher-dimensional quadrature discrepancies reported explicitly. See [degree-refinement evidence](docs/PENDULUM_REFINEMENT.md).

## Install and quick start

Python 3.12 was used for this release verification. In an isolated environment:

    python -m pip install -r requirements-tested.txt
    python examples/quick_start.py
    python scripts/verify_release.py

The core modules are in evidence/solver_snapshot/code/. They retain the numerical implementation used for the manuscript; this release adds portable entry points, documentation and verification.

    import sys
    sys.path.insert(0, "evidence/solver_snapshot/code")
    import numpy as np
    import hwc
    P, Eq = hwc.Poly, hwc.Equation
    equation = Eq(0.0, P([[(-1.0, (1,))]]), P([[(1.0, (0,))]]))
    solution = hwc.solve(equation, H=0.75, T=2.0, J=17, p=1,
                         times=np.array([0.5, 1.0, 2.0]))
    print(solution.mean)
    print(solution.variance())

## Reproduction commands

| Purpose | Command | Scope |
|---|---|---|
| Fast verification | python scripts/verify_release.py | Hashes, OU, Hermite algebra, refinement records |
| Pendulum refinement verification | python scripts/verify_pendulum_refinement.py | Coefficient/moment checks, small solver rerun, quadrature diagnostics and table regeneration |
| Current figures | python scripts/make_figures_v5.py | Replot stored data to PDF/SVG/PNG, no new MC |
| Coefficient suite | python scripts/run_campaign.py examples | Original example driver, isolated runs directory |
| Other campaigns | python scripts/run_campaign.py cubic | Also oscillator, bench, nonlinear |
| Benchmark tables | python scripts/run_campaign.py tables | Principal benchmark TeX tables from stored inputs |
| Cubic refinement | python evidence/cubic_scaling/run_cubic_scaling.py | Additional Galerkin solves |
| Dense histories | python evidence/oscillator_time/run_osc_time.py | Duffing/pendulum solver and MC histories |
| Coupled grid check | python evidence/pendulum_refinement/run_pendulum_refinement.py --output runs/pendulum_refinement.json | 131072 coupled paths, H=0.9, 1000/2000/4000 steps |

Large campaigns can be expensive. The bench driver recomputes solver timings but reuses archived MC timing references; the oscillator driver reuses archived independent covariance quadrature. Hardware timings are not portable accuracy claims. See [reproduction details](docs/REPRODUCTION.md).

## Contents

- evidence/solver_snapshot/code: solver, basis, algebra, references and experiment drivers.
- evidence/solver_snapshot/reference_results: frozen manuscript benchmark records.
- evidence/data: stored inputs/results for current plots and drivers.
- evidence/pendulum_refinement: paired grid-refinement code and results.
- evidence/pendulum_degree_refinement: degree/quadrature campaigns, same-mode MC, full records and safe numeric coefficient arrays.
- figure_design/legacy_extracts: archived plotted arrays for comparison panels.
- reference_tables: manuscript table and macro snapshots.
- scripts: portable entry points and figure generation.
- docs/SOURCE_PROVENANCE.json: original hashes; MANIFEST_SHA256.json: release hashes.

The supplementary temperature panel uses derived data from the northern-hemisphere annual series in R package longmemo; see [data provenance](docs/DATA_PROVENANCE.md). Archived comparisons are distinguished from newly recomputed results.

Manuscript drafts, correspondence, reviewer reports, credentials and unrelated projects are excluded.

## Citation and reuse

GitHub offers a citation through CITATION.cff. Cite release v1.1.0 for these results; no journal DOI is assigned here.

The code is publicly accessible. A software reuse license has not yet been selected by the authors; public visibility alone does not grant an unrestricted software license. Third-party data retain their source terms.
