# CosChaos — cosine-coordinate fractional Wiener chaos solver

**Current release: [v1.2.0](https://github.com/EslamAhmedGaber/fractional-wiener-chaos-solver/releases/tag/v1.2.0)**

Code and numerical evidence accompanying the 12 November 2026 manuscript,
*CosChaos: a cosine Wiener chaos solver for stochastic differential equations driven by fractional Brownian motion*.

CosChaos constructs finite Wiener-chaos approximations for standard fractional Brownian motion,
1/2 <= H < 1, with Wick/Skorohod noise under the model assumptions stated in the manuscript.
A deterministic coefficient system produces a polynomial in Gaussian coordinates. Its coefficients
give moments, and evaluating the polynomial gives approximate paths and sampled distributional outputs.
The cosine Gram matrix is available in closed form. For affine equations with H-independent coefficients,
the correlated-coordinate coefficient solution can be reused across H. Ordinary nonlinear drift generally
requires H-dependent coefficient systems.

## Quick reproduction of the current figures and tables

```sh
python -m pip install -r requirements.txt
python reproduction/2026-11-12/scripts/make_figures.py
python reproduction/2026-11-12/scripts/make_flagship.py
python reproduction/2026-11-12/scripts/make_tables.py
python scripts/verify_current_release.py
```

These plotting commands use stored evidence, not new Monte Carlo runs. They regenerate the five main
figures and the new main/supplementary tables. The verification command also runs small numerical
checks of the current engine and experimental backend. See
[the reproduction guide](reproduction/2026-11-12/README.md) for full experiment commands and costs.

## What is included

| Location | Purpose |
|---|---|
| `solver/` | Core coefficient solver, closed-form Gram matrix and reference examples |
| `reproduction/2026-11-12/scripts/` | Current figure/table generators and inference, rare-event and selective-projection drivers |
| `reproduction/2026-11-12/evidence/` | Stored numerical records used by the current manuscript |
| `reproduction/2026-11-12/reference_tables/` | Reference generated tables for comparison |
| `scripts/`, `evidence/` | Previously released experiments, retained for reproducibility |
| `docs/RELEASE_VALIDATION_v1.2.0.md` | Tests performed for this publication and their scope |

The update includes noisy oscillator Hurst inference, ray-corrected importance sampling for cubic
models, independent rare-event replication, paired time-step checks and selective projection for a
saturating tanh drift. A tanh drift is not a trained neural network. Reported variance-reduction factors
are not wall-clock speedups. Finite chaos solutions and distributional outputs have truncation error;
sampling adds statistical uncertainty. Timings refer to the recorded workloads and hardware.

The `solver/` implementation is unchanged from v1.1.3. The selective-projection implementation is an
experimental model-specific backend under the reproduction directory; it is not a general extension
of every core-solver API. Historical directory identifiers are retained so paths and stored records remain
traceable. The old reproduction instructions are in [docs/REPRODUCING_v1.1.3.md](docs/REPRODUCING_v1.1.3.md).

## Citation and availability

Use `CITATION.cff` to cite this software version. The release ZIP contains the tagged repository and its
SHA-256 manifest. The manuscript, journal correspondence and private future-work material are not
included in this software release. No archival DOI is claimed by this release.

## Reuse and data provenance

A software reuse license has not yet been selected in this repository. Public visibility alone does not
grant an unrestricted software license. Third-party data retain their source terms; see
[docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md).
