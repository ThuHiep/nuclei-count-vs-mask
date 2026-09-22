# Ma nguon — Image-Level Counts versus Instance Masks for Nuclei Counting

Ma nguon tai lap moi bang va hinh trong bai.

## Loi va mo hinh

| file | noi dung |
|---|---|
| `dsunet.py` | DSU-Net: mang, ham dung du lieu, vong huan luyen, suy luan |
| `losses.py` | ba so hang mat mat (mat do, so dem, NLL) |
| `data_io.py` | chi muc anh, doc mat na, tai PanNuke va NuInsSeg |
| `eval_metrics.py` | R2, MAE va cac do chat luong khoang du bao |

## Bang 1 va Bang 2 — so sanh hai che do giam sat

| file | bo du lieu |
|---|---|
| `parity_pannuke.py` | PanNuke (doi `--backbone` de sinh Bang 2) |
| `parity_nuinsseg.py` | NuInsSeg |
| `parity_conic.py` | CoNIC |
| `equivalence.py` | kiem dinh TOST, khoang bootstrap, delta* |

## Bang 3, Hinh 3, Hinh 4 — bang chung mo hinh dem tu chinh cac nhan

| file | noi dung |
|---|---|
| `deletion_doseresp.py` | xoa nhan co doi chung gia, sinh .npz |
| `fig_agreement_doseresp.py` | dung Hinh 3 va Hinh 4 tu .npz |
| `localisation.py` | do muc dinh vi cua ban do mat do |
| `localisation_from_ckpt.py` | nhu tren, nhung nap diem kiem tra da luu |

## Bang 4 — so voi cac phuong phap dung nhan vi tri

`baseline_stardist.py`, `baseline_stardist_nuinsseg.py`, `baseline_dualunet.py`,
`baseline_denuc.py`, `baseline_cellvit.py`, `baseline_cellpose.py`,
`baseline_instanseg.py`, `baseline_heavy_eval.py`, `baseline_aggregate.py`

## Bang 5 va Phu luc A

| file | noi dung |
|---|---|
| `per_organ_error.py` | sai so dem theo tung loai mo |
| `predict_dump.py` | huan luyen roi sinh file du doan (mu, sigma) |
| `predict_from_ckpt.py` | sinh du doan tu diem kiem tra da luu, khong huan luyen lai |
| `prediction_intervals.py` | khoang du bao bang split conformal |
| `per_organ_coverage.py` | do phu khoang theo tung co quan |

## Cac muc khac

| file | muc |
|---|---|
| `transfer.py` | chuyen mien giua hai bo du lieu |
| `annotation_budget.py` | duong cong ngan sach chu thich |
| `cost_params_gmacs.py`, `cost_latency.py`, `cost_profile.py` | chi phi tinh toan |
| `fig_design_patches.py`, `fig_design_grads.py`, `fig_design_pred.py`, `fig_design_make.py` | Hinh 1 |
| `fig_equivalence.py` | Hinh 2 |

## Chuan bi du lieu va kiem tra

`prep_pannuke_testfold.py`, `prep_nuinsseg_as_pannuke.py`, `prep_dualunet_folds.py`,
`precompute_pannuke_counts.py`, `test_dsunet.py`, `test_losses.py`, `verify_numbers.py`

## Bien moi truong

Cac script khong nhan duong dan qua tham so thi doc tu bien moi truong (trong ngoac la mac dinh):

| bien | dung cho | mac dinh |
|---|---|---|
| `COCO_FOLD3` | thu muc COCO cua fold kiem tra | `data/coco/fold3` |
| `FIG_DIR` | noi ghi hinh | `figures` |
| `CKPT_MASK`, `CKPT_COUNT` | diem kiem tra cho Hinh 1 | `ckpt/mask_s42.pt`, `ckpt/count_s42.pt` |
| `DATA_DIR` | thu muc chua cac file du doan | `data` |

```
COCO_FOLD3=~/data/coco/fold3 FIG_DIR=figures python fig_design_patches.py
```

## Chay thu khong can GPU

```
python test_losses.py     # 17 phep kiem ham mat mat
python test_dsunet.py     # 16 phep kiem toan tuyen tren du lieu tong hop
```

`lib/` chua hai module phu tro (`conformal.py`, `pannuke_loader.py`).

## Trong so mo hinh

Kho ma nguon nay chi chua ma nguon. Cac diem kiem tra (.pt) duoc phat hanh rieng; moi script can
den chung deu nhan duong dan qua tham so `--ckpt`.
