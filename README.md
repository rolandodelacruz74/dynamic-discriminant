# Information-dependent calibration in dynamic prediction

Code, synthetic data and results for

> R. de la Cruz (2026). *Information-dependent calibration in dynamic prediction of a binary outcome from
> longitudinal markers.* Submitted.

The repository reproduces the numerical checks of the theory (Section 3), the simulation study (Section 5),
both applications (Section 6), Tables 1 to 4, Figures 1 and 2, and Web Tables 1 to 9 of the Supporting
Information.

## Installation

```
conda env create -f environment.yml && conda activate dyndisc      # or: pip install -r requirements.txt
```

The results were obtained with Python 3.13 and the package versions pinned in `environment.yml`. PyMC is used
for posterior sampling (No-U-Turn sampler, four chains). Installing PyMC through conda-forge gives a
BLAS-linked PyTensor and much faster sampling than a pip installation.

## What each script reproduces

| Script | Paper | Running time (2 cores) |
|---|---|---|
| `scripts/01_theory_checks.py` | Theorem 1, Corollaries 2 and 3, Proposition 2 example | < 1 min |
| `scripts/02_simulation.py`, `02b_simulation_tables.py` | Section 5, Tables 1 and 2, Web Tables 1 to 6 | about 8 h (`--quick`: a few minutes; `--resume` continues an interrupted run) |
| `scripts/03_gtn.py` | Section 6.1, leave-one-out predictions of the discriminant and joint models | about 10 min |
| `scripts/03b_gtn_two_stage.py` | Section 6.1, two-stage approach | < 1 min |
| `scripts/04_pregnancy.py` | Section 6.2, fits, leave-one-out predictions with exact refits, LPML | about 3 h |
| `scripts/07_nested_cv_tempering.py gtn` | Section 6.1, nested cross-validation of tempering | about 30 min (resumable) |
| `scripts/07_nested_cv_tempering.py pregnancy --draws 2000 --tune 2000 --target-accept 0.99` | Section 6.2, nested cross-validation of tempering | about 3.5 h (resumable) |
| `scripts/08_assessment.py` | Tables 3 and 4, Figure 1, Web Tables 7 to 9 (bootstrap intervals, paired differences) | about 5 min |
| `scripts/05_figure_patients.py` | Figure 2 | < 1 min |
| `scripts/06_missingness.py` | Web Appendix E, value of the missingness pattern | < 1 min |
| `scripts/09_jmq_quadrature_check.py --reps 10` and `--regime-m --reps 10` | Web Appendix D, accuracy of the joint-model quadrature in the simulation | about 1 h each |
| `scripts/10_jm_predictive_check.py` | Section 4.1, joint-model predictive probabilities with and without weighting the posterior draws by the likelihood of the subject's measurements | about 30 min |
| `scripts/11_probit_check.py` | Web Appendix C, probit approximation versus quadrature for the joint models | about 30 min |
| `scripts/00a_generating_parameters.py`, `00b_make_synthetic.py` | synthetic data sets | < 1 min |

`make quick` runs the main steps with short settings on the synthetic data; `make all` runs the full
analysis and `make checks` the numerical checks. Scripts 03 to 08, 10 and 11 use the original data when they
are present in `data/` (see below) and otherwise fall back to the synthetic data, writing to
`results/gtn_synthetic/` and `results/pregnancy_synthetic/`; scripts 06, 10 and 11 need the original data.

## Data

See `data/README.md`. In short:

* **GTN data** (Dandis et al., 2020, Biometrical Journal 62, 398-413): not redistributed here. Download
  `hCG_GTN.csv` from the supporting information of that article and place it at `data/gtn/hCG_GTN.csv`.
  The data originate from the Dutch Central Registry for hydatidiform moles at Radboud University Medical
  Center, Nijmegen.
* **Pregnancy data**: cannot be shared publicly; available from the author on reasonable request, subject to
  the conditions of the clinic where they were collected.
* **Synthetic data** with the structure of both data sets are included, so that every script runs. Results
  obtained with them differ from those in the paper.

## Results included

`results/` contains the outputs of the runs reported in the paper:

* posterior draws of the population parameters, posterior summaries and convergence diagnostics
  (divergences, R-hat, effective sample sizes, E-BFMI) for every fit;
* for the GTN data, leave-one-out predictions, the folds of the nested cross-validation and the two-stage
  results;
* bootstrap assessments and paired differences for both applications (`assessment_*.csv`,
  `paired_differences.csv`), LPML and Pareto shape summaries, and the calibration curves (Figure 1);
* the simulation results, one row per replicate, landmark and method (`results/simulation/`), the Bayesian
  refits used to check the maximum likelihood fits (`bayes_check.csv`) and the quadrature checks
  (`results/jmq_check/`);
* summaries of the checks of Web Appendices C to E (`missingness.json`, `jm_predictive_check.json`,
  `probit_check.json`).

Subject-level outputs of the pregnancy data (leave-one-out predictions, exact refits, cross-validation folds)
and Figure 2, which shows individual measurements, are not distributed, because the data cannot be shared.
They are regenerated by the scripts when the original file is available.

## Computational notes

* In the random-inflection models the asymptote and estradiol random effects are integrated out analytically
  given the inflection time (conditional linearity). The inflection time is sampled when fitting and
  integrated on a 281-point grid when predicting.
* Predictions average over the posterior distribution: class-conditional densities are averaged over the
  draws before Bayes' theorem is applied (discriminant models), and conditional class probabilities are
  averaged with weights proportional to the likelihood of the subject's measurements (joint models).
* Leave-one-out predictions use Pareto-smoothed importance sampling from a single fit; subjects with Pareto
  shape estimate above 0.7 are refitted exactly.
* Tempering is assessed by nested cross-validation: in each of five outer folds the model is refitted to the
  other folds, the tempering coefficients are estimated from leave-one-out logits of those training subjects,
  and both are applied to the held-out fold. Output files keep the prefix `external_cv`.
* Seeds are fixed in every script.

## Citation

See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). Synthetic data and results: CC BY 4.0.

## Contact

Rolando de la Cruz, Faculty of Engineering and Sciences, Universidad Adolfo Ibáñez, Chile
(rolando.delacruz@uai.cl).
