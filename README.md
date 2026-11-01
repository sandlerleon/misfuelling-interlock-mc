# interlock_mc — barrier-integrity Monte Carlo for a laser-scanned aircraft-fuelling permissive interlock

Leon Sandler, Independent Researcher — leonsandler@alumni.swinburne.edu — ORCID 0009-0007-4584-808X

Code, tests, results and manuscript for *"A fail-safe permissive interlock against aircraft misfuelling: system concept, barrier-degradation model and
Monte Carlo risk assessment,"* prepared for **The Aeronautical Journal** (Cambridge University Press, Research Article).

Archived on Zenodo: software doi:[10.5281/zenodo.23861809](https://doi.org/10.5281/zenodo.23861809) (all versions; version 2.0.0 is
10.5281/zenodo.23861810), manuscript preprint doi:[10.5281/zenodo.23861807](https://doi.org/10.5281/zenodo.23861807) (all versions).

## What it does

Single-file, MyUncle-style framework (Python >= 3.10; numpy, scipy, matplotlib) with three levels that cross-check one another:

* **Closed-form event hazard** — event tree with stress-dependent human error, beta-factor common cause (bounded by the Fréchet–Hoeffding limits),
  relay probability of failure on demand and software fault; four barrier configurations S0 (administrative checks only), S1 (interlock as first
  conceived, 90-day audit), S2 (S1 + automated tamper analytics) and S3 (S2 + redundant tags, 1oo2 relay, inline product sensor).
* **Mean-field barrier compromise** — bypass and habitual override after nuisance blocks, with threshold-like peer feedback; the bistable window and
  hysteresis of the compromised fraction.
* **Fleet Monte Carlo** — trucks x days with Rao–Blackwellised event hazards, paired parameter draws, parameter and stochastic uncertainty separated
  (1,000 parameter sets x 5 fleet-years = 5,000 replicates), PRCC sensitivity, shock scenarios, and a crude Bernoulli event simulator that validates the
  closed form.
* **Controller state machine** and a **virtual fault-injection campaign** (nine fault categories, defences switched on and off one at a time).
* **Robustness experiments** — dependence structure, product-sensor detection probability, prior family, feedback / restoration phase diagram, convergence.
* **Break-even economics** — consequence cost per unsafe delivery over error propensity and exposure.

**All parameters are ASSUMED ideation-stage (TRL 1-3) priors, not field data. No hardware has been built. Every result is a model-conditional estimate,
not demonstrated safety performance.**

## Run

```
pip install -r requirements.txt
python interlock_mc.py --demo                          # quick self-test
python interlock_mc.py --reproduce --out out --seed 7  # all figures + results.json (about 2.5 minutes)
python tests/test_interlock_mc.py                      # 11 tests
```

Software versions and the seed are written to `results.json`.

## Reproduction check

The published `figures/results.json` was produced with Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.8 on Linux. Re-running
`--reproduce --seed 7` from a clean copy with Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.1 on Windows 10 reproduced all 1,201 numerical
entries of `results.json` exactly (only the four version/platform entries differ), and all 11 tests pass.

## Layout

```
interlock_mc.py          model, Monte Carlo, state machine, fault injection, robustness experiments, economics, figures
tests/                   11 tests (fail-safe truth table, closed form vs crude MC, state machine, linear limit, fault injection, ...)
figures/                 fig1-fig13 and results.json as used in the manuscript
manuscript/              Misfuelling_Interlock_Aeronautical_Journal_Manuscript_v3.docx / .pdf, Misfuelling_Interlock_Cover_Letter.docx
```

## Versions

* **v2.0.1** (2026-11-01) — manuscript language edit (no space before %, 'approximately', plain verbs; the journal's British spelling and spaced en
  dashes kept); concept DOIs cited in the manuscript and cover letter. Code unchanged.
* **v2.0.0** (2026-11-01) — first public release. Start and continuous-flow permission separated (Table 1, Fig. 2); fault tree and failure-modes table;
  event-tree derivation of the common-cause hazard and dependence experiments; 5,000-replicate ensembles with convergence, Wilson intervals, variance
  decomposition and prior-robustness; feedback / restoration phase diagram; virtual fault-injection campaign; break-even frontier.

## Licence

Code: MIT (LICENSE). Manuscript, figures and data: CC BY 4.0.
