# EGT training, testing, and explainability code

This release provides the code required to train the enhanced genomic transformer (EGT), evaluate it on the forward-validation test set, and run the biological interpretability analyses reported in the manuscript. The processed model-input archive must first be downloaded and extracted so that its root directory contains `PIC`, `HZA`, and `BloodLipid`.

Install the core dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Unified release entry point

`src/run_release_pipeline.py` is the recommended entry point. It calls `src/egt.py` for model fitting, validation-based checkpoint selection, and test-set evaluation. The resulting directory contains `best_checkpoint.pt`, `summary_metrics.csv`, `metrics_all.json`, predictions, the training history, and `release_run_manifest.json`.

For example, the following command trains and tests a single-trait EGT for the BloodLipid LDL-C trait, then runs integrated gradients and masking:

```powershell
python src/run_release_pipeline.py --data-root EGT_model_inputs_v1 --dataset BloodLipid --mode single --trait LDL-C --output-dir results/bloodlipid_ldlc --explain
```

For multi-task EGT, omit `--explain`. If `--traits` is omitted, the model uses all traits listed in the selected dataset's `labels_long.csv`:

```powershell
python src/run_release_pipeline.py --data-root EGT_model_inputs_v1 --dataset BloodLipid --mode mtl --output-dir results/bloodlipid_mtl
```

To reproduce the manuscript-style batch runs with the dataset-specific settings defined in the code, use:

```powershell
python src/run_all_trait_egt.py --data-root EGT_model_inputs_v1 --output-root results/egt_all --device cuda
```

## Included modules

| File | Role |
| --- | --- |
| `src/run_release_pipeline.py` | Unified command-line entry point for released inputs; orchestrates training, test evaluation, and optional explainability. |
| `src/egt.py` | EGT architecture, SNP imputation and standardization fitted on the training set, autoencoder pretraining, model training, validation checkpoint selection, test-set metrics, prediction export, and checkpoint serialization. |
| `src/explain_egt.py` | Loads a single-trait checkpoint and produces integrated-gradient SNP importance, aggregated genomic-window importance, and masking-validation results. |
| `src/baselines.py` | Shared data-loading, imputation, standardization, and metric utilities used by `egt.py`. |
| `src/preprocess.py` | Rebuilds the processed inputs and forward-validation splits from the original source datasets; it is not needed when using the released input archive. |
| `src/run_all_trait_egt.py` | Batch runner for the manuscript's single-trait and multi-task experiments over all three datasets. Use `--data-root` to point it to the extracted released inputs. |

## Interpretability outputs

When `--explain` is used, the `explainability` directory contains `attributions_test.npy`, `snp_importance.csv`, `window_importance.csv`, `masking_validation.csv`, and `explainability_manifest.json`. Integrated gradients are evaluated on the forward-validation test animals. The masking analysis compares high-importance, low-importance, and randomly selected genomic windows at the requested masking fractions.

The current pipeline supports interpretability analysis for a single-trait EGT checkpoint. Multi-task EGT training and test evaluation are supported by the unified entry point, but a multi-task checkpoint is not passed to `explain_egt.py`.
