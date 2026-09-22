#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from data_io import load_pannuke_data
from eval_metrics import eval_r2_mae
from dsunet import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", required=True)
    ap.add_argument("--train_folds", default="1,2")
    ap.add_argument("--test_fold", default="3")
    ap.add_argument("--exclude_tissue", default="", help='headline = "" (full fold3, GIỮ colon)')
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--fracs", default="0.05,0.1,0.25,0.5,1.0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_folds = [int(x) for x in args.train_folds.split(",")]
    data = load_pannuke_data(args.pannuke_root, tr_folds, args.exclude_tissue, 0)
    n_tr = len(data)
    data += load_pannuke_data(args.pannuke_root, [int(args.test_fold)], args.exclude_tissue, 0)
    all_tr = list(range(n_tr)); te = list(range(n_tr, len(data)))
    print(f"[data] train pool fold{tr_folds}={n_tr} / test fold{args.test_fold}={len(te)} "
          f"(exclude='{args.exclude_tissue or 'none'}')")

    fracs = [float(f) for f in args.fracs.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for fr in fracs:
        r2s, maes = [], []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            k = max(1, int(round(n_tr * fr)))
            tr = list(rng.choice(all_tr, k, replace=False))
            np.random.seed(sd); torch.manual_seed(sd)
            model = train(data, dev, args.epochs, 32, 1e-3, tr,
                          0.0, 1.0, 0.01, 0.5, 16, True, "poisson", args.backbone)
            r2, mae = eval_r2_mae(model, data, te, dev)
            r2s.append(r2); maes.append(mae)
        rows.append((fr, k, np.mean(r2s), np.std(r2s), np.mean(maes), np.std(maes)))
        print(f"  frac {fr:5.0%}  n={k:4d}  R² {np.mean(r2s):+.3f}±{np.std(r2s):.3f}  "
              f"MAE {np.mean(maes):.2f}±{np.std(maes):.2f}")

    print(f"\n=== LEARNING CURVE efflite0 count-only ({len(seeds)} seed) ===")
    print(f"  {'% nhãn':>7s} {'#ảnh':>6s} {'R² ':>16s} {'MAE ':>14s}")
    for fr, k, r2m, r2s, mm, ms in rows:
        print(f"  {fr:7.0%} {k:6d} {r2m:+.3f}±{r2s:.3f}   {mm:6.2f}±{ms:.2f}")
    print("\nR² bão hoà ở bao nhiêu phần trăm nhãn đếm.")


if __name__ == "__main__":
    main()
