# Configurations

`best_configs.csv` holds the configuration selected for every method and
dataset (validation Pareto filtering followed by the achievement scalarizing
function of Appendix A.2, weights 0.5/0.3/0.1/0.1).  The paper's results were
produced with these configurations; `Config_Key` names the hyperparameters in
the same form as the result files, e.g. `lambda_fair_0.3_beta_5.0_lam_0.5`
(CAFE-In: fairness penalty, RA fusion weight, mobility boost) or
`preproc_alpha_1.0_beta_0.5_lam_0.5` (CAFE-Pre).

`experiments/configs/legacy_best_configs_selected.csv` is the same selection
in the column layout read by `experiments/run_k_sensitivity.py` and
`experiments/run_stratified_cafe.py`; `experiments/configs/verify_legacy_best_configs.py`
checks that the two agree.
