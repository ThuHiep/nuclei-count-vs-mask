# Source code — Image-Level Counts versus Instance Masks for Nuclei Counting

Source code to reproduce every table and figure in the paper.

## Core model and losses

| file | content |
|---|---|
| `dsunet.py` | DSU-Net: network, dataset, training loop, inference |
| `losses.py` | the three loss terms (density, count, NLL) |
| `data_io.py` | image indexing, mask reading, PanNuke/NuInsSeg loading |
| `eval_metrics.py` | R2, MAE, and prediction-interval quality metrics |

## Table 1 and Table 2 — comparing the two supervision regimes

| file | dataset |
|---|---|
| `parity_pannuke.py` | PanNuke (pass `--backbone` to generate Table 2) |
| `parity_nuinsseg.py` | NuInsSeg |
| `parity_conic.py` | CoNIC |
| `equivalence.py` | TOST test, bootstrap interval, delta* |

## Table 3, Figure 3, Figure 4 — evidence that the count-only model uses nuclear information

| file | content |
|---|---|
| `deletion_doseresp.py` | nucleus deletion with sham control, produces .npz |
| `fig_agreement_doseresp.py` | renders Figure 3 and Figure 4 from the .npz |
| `localisation.py` | measures localisation of the density map |
| `localisation_from_ckpt.py` | same, but loads a saved checkpoint |

## Table 4 — comparison with methods trained on positional labels

`baseline_stardist.py`, `baseline_stardist_nuinsseg.py`, `baseline_dualunet.py`,
`baseline_denuc.py`, `baseline_cellvit.py`, `baseline_cellpose.py`,
`baseline_instanseg.py`, `baseline_heavy_eval.py`, `baseline_aggregate.py`

## Table 5 and Appendix A

| file | content |
|---|---|
| `per_organ_error.py` | counting error broken down by tissue type |
| `predict_dump.py` | trains then dumps predictions (mu, sigma) |
| `predict_from_ckpt.py` | generates predictions from a saved checkpoint, no retraining |
| `prediction_intervals.py` | prediction intervals via split conformal prediction |
| `per_organ_coverage.py` | interval coverage broken down by tissue type |

## Other sections

| file | section |
|---|---|
| `transfer.py` | domain transfer between the two datasets |
| `annotation_budget.py` | annotation-budget curve |
| `cost_params_gmacs.py`, `cost_latency.py`, `cost_profile.py` | computational cost |
| `fig_design_patches.py`, `fig_design_grads.py`, `fig_design_pred.py`, `fig_design_make.py` | Figure 1 |
| `fig_equivalence.py` | Figure 2 |

## Data preparation and tests

`prep_pannuke_testfold.py`, `prep_nuinsseg_as_pannuke.py`, `prep_dualunet_folds.py`,
`precompute_pannuke_counts.py`, `test_dsunet.py`, `test_losses.py`, `verify_numbers.py`

## Environment variables

Scripts that do not take a path as an argument read it from an environment variable
(default shown in parentheses):

| variable | used for | default |
|---|---|---|
| `COCO_FOLD3` | COCO directory of the test fold | `data/coco/fold3` |
| `FIG_DIR` | output directory for figures | `figures` |
| `CKPT_MASK`, `CKPT_COUNT` | checkpoints for Figure 1 | `ckpt/mask_s42.pt`, `ckpt/count_s42.pt` |
| `DATA_DIR` | directory containing prediction files | `data` |

```
COCO_FOLD3=~/data/coco/fold3 FIG_DIR=figures python fig_design_patches.py
```

## Quick tests, no GPU required

```
python test_losses.py     # 17 unit tests for the loss functions
python test_dsunet.py     # 16 end-to-end tests on synthetic data
```

`lib/` contains two helper modules (`conformal.py`, `pannuke_loader.py`).

## Model weights

This repository contains source code only. Trained checkpoints (.pt) are released
separately; every script that needs one accepts a path via the `--ckpt` argument.
