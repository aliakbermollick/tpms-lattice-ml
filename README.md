# TPMS lattice structure–property framework — data and code

Code and data for *"Machine-learning-enabled computational–experimental framework for structure–property
mapping in TPMS lattice structures"* (Mollick et al.).

| Folder | Paper section | What it does |
|---|---|---|
| `data/` | 2.3 | 270 FE results (5 TPMS families × 54 designs), `TPMS_data_27.12.25.xlsx` |
| `01_statistical_analysis/` | 3.2, Figs S1–S4 | Dataset statistics and histograms/box plots |
| `02_correlation_analysis/` | 2.4, Fig S5 | Family-wise Spearman correlation matrices |
| `03_ml_models/` | 2.5, 3.3, Table 3, Table S1 | 6 base models + Stacked Ensemble (training, and re-scoring of saved models) |
| `04_prediction_inverse_design/` | 2.6–2.7, 3.4–3.5, Tables 4–7 | Out-of-sample prediction and inverse design |
| `05_shap/` | 2.8, 3.6, Fig 4, S15–S17 | Family-wise KernelSHAP for SE and GPR |
| `06_bayesian_optimization/` | 2.9, 3.7, Tables 8–9, S2–S5 | Bayesian optimization (50 seeds) and the Jakar HPC job |
| `docs/` | - | Browser calculator for Young's modulus (Stacked Ensemble), served by GitHub Pages |

## Online calculator
Enter the TPMS family, the unit cell sizes UCx, UCy, UCz (2–4 mm) and the wall thickness t (0.30–0.40 mm) to get the
predicted Young's modulus. The model runs in your browser through Pyodide (Python in WebAssembly), so nothing is sent to
a server. The first load downloads Python and scikit-learn and takes about 20–40 seconds.
Live page: https://aliakbermollick.github.io/tpms-lattice-ml/docs/

Each folder has `reference_outputs/` or `reference_results/` with the numbers from the runs used in the paper.

## Environment
The saved models were pickled with **scikit-learn 1.6.1 / numpy 2.x** and load with scikit-learn 1.5.2–1.6.1 (not 1.9).
BO used scikit-optimize 0.10.2. Pickled models are not stored in this repo (about 50 MB); re-train with
`03_ml_models/train_models.py` or ask the corresponding author. Set `TPMS_MODEL_RUN` to the model folder.

## Verification summary (reproduction of the paper's numbers)
- Sections 3.2 statistics, Spearman matrices, Table 3, Tables 4–5: reproduce exactly.
- SHAP (both models): recomputed tables identical to the saved ones.
- BO on Jakar (40 cores, 360 GB): GPR 11.7 min, SE 21.7 min, against 88.5 and 162.2 min on 5 workers.
  Per campaign, 579/600 (GPR) and 585/600 (SE) best values are identical to the original runs; the best
  over 50 seeds is identical (E 802.11 GPR / 795.86 SE; Especific 1.18927 GPR / 1.19086 SE).
  The remaining campaigns diverge late in the search, likely from library-version differences.

## Notes on the published tables
- **Tables 6–7 (inverse design):** the search returns 30 finalists per model and family. One candidate per
  family was selected from these for FE validation; it is not always the minimum-error finalist.
- **Table 6, Gyroid:** the code gives 248.67 MPa (the paper lists 248.70).
- **Table 8 / Sec. 3.7.1:** the SE global optimum for specific stiffness is 1.1909 (Table 9, `bo_best_summaries_jakar_SE.csv`);
  the text lists 1.181.
- **Fig 4 / Sec. 3.6:** UCz is the top driver of specific stiffness in Diamond, Gyroid, Lidinoid and SplitP; in Schwarz
  UCy and UCx are larger.
- **Sec. 3.2:** Spearman ρ between wall thickness and E is 0.33–0.53 (moderate).
- **Table S1** does not list the GPR search (`alpha` ∈ {1e-6, 1e-5, 1e-4}, `n_restarts_optimizer` ∈ {0, 2, 5};
  kernel = Constant × ARD-RBF + White). CV was 5-fold, stratified by family; the test set is 54 designs (20%).
- **Figures S3/S4:** `01_statistical_analysis/statistical_analysis.py` regenerates equivalent figures; the original
  figure script was not found.

## Data availability
FE inputs and outputs: `data/`. Experimental data (compression tests, dimensions, FTIR) are available from the
corresponding author on request.

## License
The code and data in this repository are released under the MIT License (see `LICENSE`).

## Citation
If you use this repository, please cite it using `CITATION.cff` (GitHub's "Cite this repository" button) and the
accompanying paper once it is published. A Zenodo archive with a DOI is planned; the DOI will be added here after
the first release.
