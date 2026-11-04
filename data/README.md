# Data

## Pregnancy outcome after assisted reproduction (Section 6.2)

The original data (161 women, 348 visits, log10 beta-HCG and estradiol) were collected at a private assisted
reproduction clinic in Santiago, Chile, and were analyzed by Marshall et al. (2006, 2009). The author is not
authorised to distribute them. They are available from the author on reasonable request
(rolando.delacruz@uai.cl), subject to the conditions of the clinic.

`pregnancy/pregnancy_synthetic.txt` is a synthetic data set with the structure of the original data: 124 normal
and 37 abnormal pregnancies, the distribution of the number and timing of visits and the missingness rates of
each class, and values simulated from the fitted random-inflection discriminant model of each class. Columns:
`Order Patient Time SubUnit Estradiol Class` (time in days of gestation; SubUnit = log10 beta-HCG;
Estradiol = log10 estradiol; NA = missing; Class 0 = normal, 1 = abnormal). The original file, if available,
should be placed at `pregnancy/datos.txt` (same columns, without `Class`; patients 1-124 are normal).

## Post-molar gestational trophoblastic neoplasia (Section 6.1)

The data were analyzed by Dandis et al. (2020, Biometrical Journal 62, 398-413,
https://doi.org/10.1002/bimj.201900044) and are distributed with the supporting information of that article
(file `hCG_GTN.csv` in "Code and Data"). Their source is the Dutch Central Registry for hydatidiform moles at
Radboud University Medical Center, Nijmegen, The Netherlands. They are not redistributed here: download them
from the supporting information of Dandis et al. (2020) and place the file at `gtn/hCG_GTN.csv`.

`gtn/gtn_synthetic.csv` is a synthetic data set with the same structure (299 women without and 140 with GTN,
weeks 2-7, missingness rates by class and week) and values simulated from the fitted class-specific linear
mixed models. Columns: `id, weeknr, loghcg, PTD` (999 = missing; PTD = 1 for GTN).

## Generating parameters

`synthetic_parameters.json` contains the posterior means and aggregate frequencies used to generate both
synthetic data sets (`scripts/00b_make_synthetic.py`). It was produced from the original data by
`scripts/00a_generating_parameters.py` and contains no individual-level information.
