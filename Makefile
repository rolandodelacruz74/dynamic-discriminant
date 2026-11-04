# Reproduce the results of the paper. `make quick` runs every script on short settings (about 20 minutes).
PY ?= python

.PHONY: all quick theory simulation gtn pregnancy assessment figure missingness synthetic checks

all: theory simulation gtn pregnancy assessment figure

theory:
	$(PY) scripts/01_theory_checks.py

simulation:                      # about 8 h on 2 cores
	$(PY) scripts/02_simulation.py --jobs 2 --resume
	$(PY) scripts/02b_simulation_tables.py

gtn:                             # about 10 min; needs data/gtn/hCG_GTN.csv, otherwise uses synthetic data
	$(PY) scripts/03_gtn.py
	$(PY) scripts/03b_gtn_two_stage.py
	$(PY) scripts/07_nested_cv_tempering.py gtn

pregnancy:                       # several hours; uses synthetic data unless data/pregnancy/datos.txt is present
	$(PY) scripts/04_pregnancy.py
	$(PY) scripts/07_nested_cv_tempering.py pregnancy --draws 2000 --tune 2000 --target-accept 0.99

assessment:                      # needs the outputs of gtn and pregnancy
	$(PY) scripts/08_assessment.py

figure:
	$(PY) scripts/05_figure_patients.py

checks:                          # numerical checks of Web Appendices C and D
	$(PY) scripts/09_jmq_quadrature_check.py --reps 10
	$(PY) scripts/09_jmq_quadrature_check.py --regime-m --reps 10
	$(PY) scripts/10_jm_predictive_check.py
	$(PY) scripts/11_probit_check.py

missingness:                     # needs the original data of both applications
	$(PY) scripts/06_missingness.py

synthetic:                       # regenerate the synthetic data sets from data/synthetic_parameters.json
	$(PY) scripts/00b_make_synthetic.py

quick:
	$(PY) scripts/01_theory_checks.py
	$(PY) scripts/02_simulation.py --quick
	$(PY) scripts/02b_simulation_tables.py --quick
	$(PY) scripts/03_gtn.py --data data/gtn/gtn_synthetic.csv --quick
	$(PY) scripts/04_pregnancy.py --data data/pregnancy/pregnancy_synthetic.txt --quick
	$(PY) scripts/07_nested_cv_tempering.py gtn --quick
	$(PY) scripts/07_nested_cv_tempering.py pregnancy --quick
	$(PY) scripts/08_assessment.py --suffix _synthetic_quick --boot 200
