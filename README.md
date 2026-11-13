# Reproducibility bundle (Revision 1)

Code, data and outputs for *Selecting energy-management policies for a grid-limited residential microgrid with coupling-aware multi-period hesitant fuzzy decision analysis* (revised manuscript, November 2026).

Every number, table and figure in the revised paper and in the revised Supplementary Information is produced by `run_all.py` from the three data files in `data/`, with fixed random seeds (`SEED = 2026`).

## What changed in Revision 1

| Reviewer comment | Change in the code | Main outputs |
|---|---|---|
| R2.4, R3.3 (align tuning and evaluation) | Battery rule with two settings (grid-charging target × discharge mode); tuning under the cost objective (`A2`, `A4`) and the multi-criteria objective (`A5`, `A6`) from the same training runs (`src/pipeline.py`, `tune`) | `tables/tuning.tex`, `tuning_train.csv` |
| R2.3, R3.2 (curtailed demand, comfort) | Criterion `C6` demand-side comfort; end-use service ratio `C2_svc`; comfort-weight variants (`src/ems_model.py`, `block_criteria`) | `tables/criterion_variants.tex`, `tables/reliability_constraint.tex`, `tables/kpi.tex` |
| R2.5 (perfect foresight) | Forecast-based controllers `L2F`, `L4F`: daily LP on a day-ahead persistence forecast, executed against actual data (`forecast_dispatch`) | `tables/benchmarks.tex`, `benchmarks.csv` |
| R3.1 (conventional MCDM) | VIKOR, PROMETHEE II, min–max SAW/MABAC, TOPSIS, rank-reversal rate (`src/mcdm.py`; `section_conventional`) | `tables/conventional_mcdm.tex`, `conventional_mcdm.csv` |
| R3.4 (other region, measured load) | Second site, Sydney: measured household load and PV (`src/build_sydney_data.py`, `ems_data.build_inputs_sydney`) | `tables/load_model_rankings.tex`, `summary.json` → `sydney`, `tables/sydney_detail.tex` |
| R3.5 (joint sensitivity) | Preference map grid limit × tariff ratio with re-tuning; 20,000-draw joint Monte Carlo of the decision parameters with a classification tree (`section_preference_map`, `section_joint`) | `preference_map.csv`, `joint_mc.csv`, `joint_tree.txt`, `tables/joint_tree.tex`, `figures/fig_joint.png` |
| R2.6 (implied agreement) | Column stating which operator–method agreements follow from the theory | `tables/operators_ranks.tex` |
| R1.2 (DR reference price) | Unchanged code; `summary.json` → `pibar_*` documents the duration-weighted mean tariff used by `dr_fraction` | `summary.json` |
| R1.5 (Figure 9) | The ranks figure now shows every reported variant | `figures/fig_drivers.png` |

The Morris screening now reports the gap between the two multi-criteria-tuned storage policies and the lead of the better of them over the better policy without storage.

## Quick start

The bundle needs Python 3.10 or newer and the packages in `requirements.txt`. Install them **before** the first run; otherwise `run_all.py` stops and lists the missing packages.

**Windows (Command Prompt or PowerShell), from the bundle folder:**

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_all.py --skip-gsa
```

**Linux or macOS:**

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_all.py --skip-gsa
```

Run options:

```bash
python run_all.py --skip-gsa     # everything except the Morris runs (reuses results/gsa_morris.csv)
python run_all.py                # full run, including the 120 Morris simulations
python run_all.py --figures-only # redraw figures from results/
```

Runtime (2 CPU cores): about 10–12 min for `--skip-gsa` (the preference map and the one-at-a-time sweeps re-tune under both objectives), plus about 20 min for the Morris screening. The Morris runs are resumable: progress is written to `results/gsa_morris.csv.partial.csv` and an interrupted run continues from there. No network access is needed at run time.

**Troubleshooting.** `ModuleNotFoundError: No module named 'pvlib'` (or another package) means the requirements were not installed into the Python that runs the script. Activate the virtual environment (`.venv\Scripts\activate` on Windows) and run `python -m pip install -r requirements.txt` again; using `python -m pip` guarantees that the packages go to the same interpreter. If PowerShell refuses to run the activation script, use `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe run_all.py --skip-gsa` instead.

`ImportError: DLL load failed while importing h5: An Application Control policy has blocked this file` comes from h5py, an optional dependency of pvlib that Windows (Smart App Control or a managed security policy) refuses to load. The bundle does not use h5py: `src/ems_data.py` detects this case and loads pvlib without it.

## Data provenance

| File | Content | Source | MD5 |
|---|---|---|---|
| `data/zurich_electricity_consumption.csv` | 15-min metered low-voltage demand (ewz, Value_NE5) and weather (UGZ station Zch_Stampfenbachstrasse: global horizontal irradiance, air temperature, wind speed), 2015–2022 | City of Zurich open government data, as distributed with the darts library (`datasets/zurich_electricity_consumption.csv` at https://github.com/unit8co/darts) | `764bdd3bb4f9066ba7523accf9089505` |
| `data/household_load_15min.csv` | Stochastic load of 14 apartments, 2015–2021 | Generated by `src/gen_household_load.py` (richardsonpy, Richardson et al. 2010), driven by the Zurich irradiance; one seed per household-year | `169a096911134837c708042cbe266b1c` (richardsonpy 0.2.3) |
| `data/sydney_ausgrid_15min.csv` (new) | Sum of the metered general consumption of 14 households and median metered PV output per kWp, Sydney region, 1 July 2012 – 30 June 2013, 15-min | Derived by `src/build_sydney_data.py` from the Ausgrid solar home electricity data (file `2012-2013 Solar home electricity data v2.csv`, MD5 `f2aaa003772dbd4329935a525066d733`; Ratnam et al. 2017, doi:10.1080/14786451.2015.1100196). The original Ausgrid download page is offline; the copy used is in `Data.zip` of https://github.com/rockey-abhishek/Ausgrid-Dataset. Selected households: `data/sydney_ausgrid_15min_homes.txt` | `b4783f9c359a5065e216b6fba562b260` |

`src/ems_data.py` checks the MD5 of the Zurich and Sydney files before use. It corrects the 30-min early stamping of the hourly weather means and converts all Zurich times to local standard time (CET). The BDEW H0 comparator profile is generated at run time with demandlib. To regenerate the household load from scratch, run `python src/gen_household_load.py`; to rebuild the Sydney file, run `python src/build_sydney_data.py "<path to the Ausgrid 2012-2013 csv>"`. Both files are cached, so `run_all.py` does not regenerate them.

## Analysis design

* **Policies.** A1 renewable-first; A2 battery (cost-tuned); A3 price-elastic demand response; A4 hybrid (cost-tuned); A5 battery (multi-criteria-tuned); A6 hybrid (multi-criteria-tuned) (`src/ems_model.py`, `simulate`). The DR reference price is the duration-weighted daily mean tariff (`dr_fraction`; 0.1708 USD/kWh in the baseline).
* **Benchmarks.** Cost-optimal perfect-foresight daily LPs L2 (battery), L3 (DR), L4 (both), solved with HiGHS through SciPy (`lp_benchmark`), and forecast-based controllers L2F, L4F that solve the same LP on a day-ahead persistence forecast and execute the plan against the actual data (`forecast_dispatch`).
* **Out-of-sample protocol.** The battery-rule settings (grid-charging target off/30/45/60/75/90 % × discharge mode on-peak/any deficit) are tuned on 2015–2017 under two objectives from the same simulations (`src/pipeline.py`, `tune`), and all decision results use 2018–2021 (1,460 days). Every simulation-based sensitivity run re-tunes under both objectives. At the Sydney site the rules are tuned on alternating 28-day blocks and evaluated on the blocks in between (169 test days).
* **Criteria.** C1–C6 are computed per day and tariff period with reference-point normalization (`block_criteria`); C6 is demand-side comfort with a discomfort weight δ = 0.5 of shifted energy. Storage provenance is tracked with a well-mixed two-pool battery.
* **Dynamic hesitant fuzzy soft set.** Period-level ranking HFEs {P10, P50, P90}; observation HFEs (full daily multisets) for the coupling and risk analysis; context-dependent active criteria in `section_context`.
* **Decision analysis.** `src/mcdm.py` contains the HFE constructions; the independent, comonotone and empirical couplings with AR/EIN/ALG means and ε-capping; risk-averse scores and lower-tail means; MABAC, WSM, TOPSIS and HF-TOPSIS; conventional crisp methods (SAW, original MABAC, TOPSIS, VIKOR, PROMETHEE II) and the rank-reversal rate; SMAA; and the moving-block bootstrap.
* **Sensitivity.** One-at-a-time sweeps, Morris screening over five factors with SALib (`src/gsa.py`), a re-tuned preference map (grid limit × tariff ratio) and a joint Monte Carlo analysis of the decision parameters summarized by a depth-4 classification tree (scikit-learn).

## Layout

| Path | Content |
|---|---|
| `LICENSE` | MIT licence for the code, and the terms that apply to the input data |
| `run_all.py` | Full analysis, organized by section; writes `results/*.csv`, `results/tables/*.tex`, `results/summary.json`, `figures/*.png` |
| `src/ems_data.py` | Data loading, checksums, time alignment, PV (Erbs + PVWatts via pvlib), wind, load models, tariff, Sydney site |
| `src/gen_household_load.py` | Generator for the stochastic household load |
| `src/build_sydney_data.py` | Builder of the Sydney input file from the Ausgrid data (new) |
| `src/ems_model.py` | Dispatch rules, storage provenance, LP benchmarks, forecast-based controllers, block criteria |
| `src/pipeline.py` | Input caching, tuning under both objectives, simulation runs, baseline MABAC |
| `src/mcdm.py` | Hesitant fuzzy operators, ranking/robustness methods and conventional crisp methods |
| `src/gsa.py` | Morris screening (resumable) and preference-map evaluation |
| `src/figures.py` | All data figures |
| `results/` | Outputs as used in the paper: CSV files, `summary.json` (every number quoted in the text) and `tables/*.tex` (the paper's tables in LaTeX form) |
| `figures/` | Figures 3–10 and S1–S2 as 300-dpi PNG files |

## Where each table and figure comes from (revised numbering)

| Item | Source in `results/` (or code) |
|---|---|
| Table 1 (positioning) | written by hand from the cited literature |
| Table 2 (load models and sites) | `tables/load_models.tex`; `summary.json` → `load_models` |
| Table 3 (parameters) | `BASE` in `src/ems_model.py`; `CONFIG` in `src/ems_data.py` |
| Table 4 (tuning under both objectives) | `tables/tuning.tex`, `tuning_train.csv` |
| Table 5 (annual outcomes incl. benchmarks) | `tables/kpi.tex`, `kpi_test.csv` |
| Table 6 (scores, MABAC) | `tables/mabac.tex`, `scores.csv`, `mabac_distances.csv` |
| Table 7 (benchmarks) | `tables/benchmarks.tex`, `benchmarks.csv` |
| Table 8 (criterion variants, comfort, leave-one-out) | `tables/criterion_variants.tex`, `criterion_variants.csv` |
| Table 9 (reliability and service as constraints) | `tables/reliability_constraint.tex`, `reliability_constraint.csv` |
| Table 10 (context-dependent active criteria) | `tables/context.tex`, `context_parameters.csv` |
| Table 11 (HFE construction) | `tables/hfe_construction.tex`, `hfe_construction.csv` |
| Table 12 (risk-averse rankings) | `tables/risk_rankings.tex`, `risk_rankings.csv` |
| Table 13 (reversal thresholds β*) | `tables/beta_star.tex`, `reversal_thresholds.csv` |
| Table 14 (operators and methods, implied agreement) | `tables/operators_ranks.tex`, `operators_methods.csv` |
| Table 15 (normalization) | `tables/normalization.tex`, `normalization.csv` |
| Table 16 (criterion weights) | `tables/criterion_weights.tex`, `criterion_weights.csv` |
| Table 17 (SMAA) | `tables/smaa.tex`; `summary.json` → `smaa_flat`, `smaa_conc`, `central_weights` |
| Table 18 (conventional MCDM) | `tables/conventional_mcdm.tex`, `conventional_mcdm.csv` |
| Table 19 (bootstrap) | `tables/bootstrap.tex`, `bootstrap_pairs.csv`, `bootstrap_Q.npy` |
| Table 20 (load models and second site) | `tables/load_model_rankings.tex`, `load_model_rankings.csv` |
| Table 21 (Morris) | `tables/gsa.tex`, `gsa_morris.csv`, `gsa_indices.csv` |
| Table 22 (conditions from the joint analysis) | `tables/joint_tree.tex`, `joint_tree_leaves.csv`, `joint_tree.txt`, `joint_mc.csv` |
| Figures 1–2 (framework, flowchart) | schematics, not generated from data |
| Figures 3–10 | `figures/fig_inputs.png`, `fig_hfe.png`, `fig_scores.png`, `fig_coupling.png`, `fig_robust.png`, `fig_gsa.png`, `fig_joint.png`, `fig_drivers.png` |
| Table S1 (annual inputs) | `tables/annual_inputs.tex`, `annual_inputs.csv` |
| Table S2 (criterion correlation) | `tables/criterion_corr.tex`, `criterion_correlation.csv` |
| Tables S3–S5 (ranking HFEs) | `tables/hfe_t1_offpeak.tex`, `hfe_t2_midpeak.tex`, `hfe_t3_onpeak.tex` |
| Table S6 (coupling cells) | `tables/coupling_cells.tex`, `coupling_cells.csv` |
| Table S7 (operator scores) | `tables/operators_scores.tex`, `operators_methods.csv` |
| Table S8 (temporal weights) | `tables/temporal.tex`, `temporal_weights.csv` |
| Table S9 (per test year) | `tables/per_year.tex`, `per_year.csv` |
| Table S10 (one-at-a-time sweeps) | `tables/oat.tex`, `oat.csv` |
| Table S11 (second site) | `tables/sydney_detail.tex`; `summary.json` → `sydney` |
| Table S12 (joint analysis, conditional winner shares) | `tables/joint_conditional.tex`, `joint_conditional_*.csv` |
| Figures S1–S2 | `figures/fig_oat.png`, `figures/fig_kpi.png` |
| All numbers quoted in the text | `results/summary.json` |

## Verifying the results

To check that the bundle reproduces the paper, move `results/` and `figures/` aside, run `python run_all.py` (or `--skip-gsa` with `results/gsa_morris.csv` restored), and compare the new `results/*.csv`, `results/summary.json` and `results/tables/*.tex` with the originals. On Windows, CSV files are written with Windows line endings, so compare them by content rather than byte for byte.

## Licence

The code is released under the MIT License (see `LICENSE`), which also covers the generated `results/` and `figures/` and the generated household-load file. The Zurich input data remain City of Zurich open government data, subject to the terms of use stated by the City of Zurich. The Sydney file is derived from data published by Ausgrid for research use (Ratnam et al. 2017); users should cite that source. `LICENSE` gives the details.

## Citation

If you use this framework or code, please cite the manuscript. A BibTeX entry will be added once publication details are finalized.
