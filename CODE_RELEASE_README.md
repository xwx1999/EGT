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
| `src/preprocess_pigheat.py` | Builds the PigHeaT inputs from the public deposit: parses the released genotype and phenotype tables, restricts to autosomal markers (SSC1-SSC18), de-duplicates markers, applies the marker quality-control mask (call rate >= 0.95 and MAF >= 0.01), and constructs the birth-year forward split (2013 training/validation, 2014 test). **Added in this revision**; it was absent from earlier code packages. |
| `src/run_all_trait_egt.py` | Batch runner for the manuscript's single-trait and multi-task experiments over all three datasets. Use `--data-root` to point it to the extracted released inputs. |

## Revision analysis scripts

`scripts/` contains the analyses added for the reviewer revision. They are not
required to reproduce the original benchmark, but they regenerate every table and
figure the revision adds, and each writes machine-readable output.

| File | Role |
| --- | --- |
| `scripts/animals_round1.py` | The revision benchmark protocol: 87 validation-only selection runs, then 435 repeat runs (5 seeds x 3 datasets x 12 traits x 8 models), producing `selection_lock.json`, `repeat_metrics_*.csv` and the per-run cost records. |
| `scripts/regenerate_round1_summary.py` | Rebuilds the summary layer (mean/SD per cell, cost table, benchmark status) from the per-run records, validating completeness first. |
| `scripts/paired_bootstrap_ci.py` | Paired bootstrap confidence intervals (B = 10,000) for the EGT-versus-baseline difference in PCC, bias and regression slope, resampling both test animals and replicate seeds. |
| `scripts/dependent_correlation_test.py` | Williams' test for dependent correlations, the formal alternative to the bootstrap for comparing two correlations that share the phenotype. |
| `scripts/masking_repeats_and_controls.py` | Repeated random masking with confidence intervals plus a size-matched control and locus-scale positive controls; runs against either the original or the PC-corrected checkpoints. |
| `scripts/ssc13_local_ld.py` | Local linkage disequilibrium and allele frequency around MARC0013088 in the pooled cohort and per breed. |
| `scripts/ssc13_structure_test.py` | Principal component analysis of the pooled genotypes and the association of the candidate marker with the leading components. |
| `scripts/ssc13_qtldb_query.py` | Extracts the SSC13 records from the Animal QTLdb legacy deposition. |
| `scripts/yang2015_ssc13_loci.py` | Extracts the SSC13 loci from the supplementary tables of the published GWAS on the same animals. |
| `scripts/audit_response_coverage.py` | Checks that every reviewer comment has a response and flags any still written as a future promise. |

## Interpretability outputs

When `--explain` is used, the `explainability` directory contains `attributions_test.npy`, `snp_importance.csv`, `window_importance.csv`, `masking_validation.csv`, and `explainability_manifest.json`. Integrated gradients are evaluated on the forward-validation test animals. The masking analysis compares high-importance, low-importance, and randomly selected genomic windows at the requested masking fractions.

The current pipeline supports interpretability analysis for a single-trait EGT checkpoint. Multi-task EGT training and test evaluation are supported by the unified entry point, but a multi-task checkpoint is not passed to `explain_egt.py`.
