# Shared Gamma Multi-Task Learning for Health Insurance Occurrence and Cost Prediction

Minimal code and frozen results accompanying the **accepted full paper, IEEE SSCI 2027 (CIHM)**. These are the essential research files, not a full archive of every exploratory experiment.

## Contents

- `paper_code/unified_m3_paper_ablations.py` — available script variant for the development ablations (Table II); **not verified as byte-identical** to the original script used for the frozen table.
- `paper_code/gamma_mtl_audit_m4_final_FIXED.py` — M4 feature engineering, training and model helpers.
- `paper_code/final_m4_fair_comparison_explainability.py` — final comparison, bootstrap, efficiency and explainability workflow. **SHA-256 matches the final source recorded in the accepted paper artifacts.** Requires the M4 helper file beside it.
- `results/development_ablation_summary.csv` — frozen development ablations (Table II).
- `results/final_oot_model_comparison.csv` — frozen 2018–2019 model comparison (Table III).
- `results/final_oot_policy_bootstrap.csv` — frozen policy-level bootstrap intervals for the reported comparisons.
- `results/final_complexity.csv` — frozen runtime, parameter and disk-size results (Table IV).
- `results/final_explainability_stability_fidelity.csv` and `results/final_ig_shap_agreement.csv` — frozen SHAP/IG summary and deletion fidelity (Table V).
- `results/final_top_features.csv` — archived feature-level explanation values.
- `figures/accepted_figure2_task_contrast.pdf` — exact 8-feature Figure 2 shown in the accepted manuscript.
- `requirements.txt` — Python package requirements (not an exact historical environment lockfile).

## Main result

The paper reports for Gamma-MTL: ROC-AUC **0.8693**, Brier **0.1015**, expected-cost MAE **623.17**, expected-cost RMSE **1571.48**, positive-severity RMSE **1745.96**. The reported expected-cost RMSE difference versus HGB is **not statistically conclusive**.

The 2017–2018 cohort was used for development. The 2018–2019 cohort is a **frozen, post-hoc temporal comparison** (not an untouched external test). Table III reports means and standard deviations across four training seeds. Bootstrap intervals were calculated on seed-averaged predictions, with policies as the resampling unit; they **do not quantify retraining uncertainty**.

## Reproduction limits

These original scripts depend on prepared 2017–2018 and 2018–2019 parquet inputs and trained checkpoint/preprocessor outputs **not included** in this minimal release. Consequently, the package is **not claimed to be an independently verified, end-to-end reproduction**. The public source data are described in the manuscript (Lledó et al., Mendeley Data; DOI: `10.17632/386vmj2tbk.4`), but the exact data-preparation pipeline and original checkpoint files are not distributed here.

The earlier development-only Gamma-weight and GLM/HGB hyperparameter screens are **not included** in this intentionally minimal package. They were checked separately against stored historical artifacts; their earlier split must not be confused with the final Table II split. No new results have been generated or substituted.

The scripts are provided **unchanged**, for archival transparency. Running them may need the original project directory structure and compatible software versions. Use `python -m py_compile paper_code/*.py` for a syntax check; this does not validate numerical reproduction.

All frozen CSVs and all three source files are included as archived, without recomputation. No private insurer records or per-policy predictions are included.
