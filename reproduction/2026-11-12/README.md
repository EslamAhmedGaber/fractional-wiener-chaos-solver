# Reproducing the 12 November 2026 CosChaos results

Run commands below from the repository root after installing `requirements.txt`.
The core solver in `solver/` is unchanged. Experiment snapshots in this directory preserve the
implementations used to generate the recorded results.

## Figures and tables from stored evidence

```sh
python reproduction/2026-11-12/scripts/make_figures.py
python reproduction/2026-11-12/scripts/make_flagship.py
python reproduction/2026-11-12/scripts/make_tables.py
python scripts/verify_current_release.py
```

Figures 2–5 go to `reproduction/2026-11-12/figures/one/` and `two/`;
Figure 1 goes to `figures/flagship/`. The plotting scripts check 40 and 8 data assertions.
Tables go to `source/tables/` and `supplement/tables/` beneath this directory. Five corresponding
reference files are supplied. Rendering metadata and fonts can vary by machine; the validation
compares numerical records and table text rather than requiring identical PDF bytes.

| Manuscript output | Evidence | Driver |
|---|---|---|
| Fig. 1, construction and outputs | Gram code, oscillator, Wick and law records | `scripts/make_flagship.py` |
| Fig. 2, Gram and noise truncation | `evidence/data/` | `scripts/make_figures.py` |
| Fig. 3, verification and work–precision | `evidence/round3_fwhe/`, `round3_full/` | `scripts/make_figures.py` |
| Fig. 4, oscillator and noisy Hurst inference | `evidence/data/v12_oscillator.json`, `round13/hurst_inference.json` | `scripts/make_figures.py` |
| Fig. 5, cubic law and rare events | `evidence/round4/`, `round5/`, `rare_events_pooled/` | `scripts/make_figures.py` |
| Main rare-event and tanh tables | `evidence/rare_events_pooled/`, `round12_1/` | `scripts/make_tables.py` |
| Supplementary candidates, replication, time steps | `evidence/round14/`, `rare_events_pooled/` | `scripts/make_tables.py` |

The dated directory and historical evidence identifiers locate calculation records, not different
mathematical methods. The main manuscript and private research plans are not part of this repository.

## Recompute experiments

Use a separate working copy for expensive reruns: experiment commands write output records and may
overwrite stored files. Seeds, grids, model coefficients and sample counts are specified in the drivers.
Timings depend on hardware and numerical libraries. Multi-million-path runs can take substantially
longer than the figure/table regeneration.

Noisy oscillator Hurst inference (pilot first, full run contains 600 data sets):

```sh
python reproduction/2026-11-12/scripts/round13_experiments/hurst_inference.py pilot
python reproduction/2026-11-12/scripts/round13_experiments/hurst_inference.py full
```

Outputs are written to `evidence/round13/`. `CC_J`, `CC_OBS`, `CC_NOBS` and `CC_MREC` permit explicit
design variations; leave defaults unchanged to reproduce the reported design. The model uses noisy
displacement observations of repeated oscillator records; this is statistical inference, not neural training.

Cubic rare events, candidates and independent double-well references:

```sh
python reproduction/2026-11-12/scripts/round14_experiments/rare_events.py ray pilot
python reproduction/2026-11-12/scripts/round14_experiments/rare_events.py ray full
python reproduction/2026-11-12/scripts/round14_experiments/rare_events.py ref 0
python reproduction/2026-11-12/scripts/round14_experiments/rare_events.py ref 1
python reproduction/2026-11-12/scripts/round14_experiments/rare_events.py ref 2
python reproduction/2026-11-12/scripts/close_experiments.py
python reproduction/2026-11-12/scripts/paired_timestep.py
```

The reference commands each use 2^23 paths. The independent replication covers 18 events with 65,536
paths each; the paired step-size study covers four events. Their outputs go to `reports/` so the
published pooled evidence is retained. The smallest corrected shift norm is an empirical selection
rule, not a theorem of globally optimal importance sampling. `scripts/pool_rare_events.py` regenerates
the published pooled CSV/JSON from the included first-run and replication records.

Selective projection for the additive tanh model:

```sh
python reproduction/2026-11-12/scripts/round12_experiments/validate_backend.py
python reproduction/2026-11-12/scripts/round12_experiments/complete_refinement.py
python reproduction/2026-11-12/scripts/rqmc_frontier.py
```

The benchmark reference is included at `scripts/fixed_model_benchmark/results/reference.json`;
recompute it with `python reproduction/2026-11-12/scripts/fixed_model_benchmark/run_benchmark.py make_reference`.
Generated selective-backend results go to `scripts/round12_experiments/results/`. Published reference
records remain in `evidence/round12_1/`. The code uses a fixed scalar saturating drift. Its path-evaluation
throughput comparison is a different workload from estimating moments to matched accuracy.

Older OU, Wick, Caputo and cubic experiment drivers remain in the top-level `scripts/` directory;
see `docs/REPRODUCING_v1.1.3.md`. Large raw sample arrays are regenerated rather than distributed.

## Verification scope

Release checks cover file integrity, table regeneration, figure-data assertions, numerical solver smoke
tests, selective-projection identities and pooled-estimator consistency. They do not replace the
mathematical proofs or certify every stochastic confidence interval. Expensive reference simulations
were not rerun solely to publish this release.
