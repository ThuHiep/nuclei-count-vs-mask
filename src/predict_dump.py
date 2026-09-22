#!/usr/bin/env python3
import argparse, os, pickle
import numpy as np
import torch
from data_io import load_pannuke_data
from dsunet import train, predict_r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", required=True)
    ap.add_argument("--train_folds", default="1,2")
    ap.add_argument("--test_fold", default="3")
    ap.add_argument("--exclude_tissue", default="", help='headline = "" (full fold3, GIỮ colon)')
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma_mode", default="poisson", choices=["poisson", "raw", "nb"],
                    help="poisson=√μ·exp(log_s) (headline); raw=exp(log_s) (ablation: sập trên dải count rộng?)")
    ap.add_argument("--out", default="work/efflite0_pan_countonly_uq_f3.pkl")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_folds = [int(x) for x in args.train_folds.split(",")]
    data = load_pannuke_data(args.pannuke_root, tr_folds, args.exclude_tissue, 0)
    n_tr = len(data)
    test_data = load_pannuke_data(args.pannuke_root, [int(args.test_fold)], args.exclude_tissue, 0)
    data += test_data
    tr = list(range(n_tr))
    print(f"[data] train fold{tr_folds}={n_tr} / test fold{args.test_fold}={len(test_data)} "
          f"(exclude='{args.exclude_tissue or 'none'}')")

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    # train(data, device, epochs, ch, lr, train_idx, w_density, w_count, w_nll, beta, bs, detach_mu, sigma_mode, backbone)
    model = train(data, dev, args.epochs, args.ch, 1e-3, tr,
                  0.0, 1.0, 0.01, 0.5, 16, True, args.sigma_mode, args.backbone)

    out = predict_r2(model, test_data, dev)          # {"preds":[{mu,sigma}], "gts":[[gt]], "organs":[org]}
    gt = np.array([g[0] for g in out["gts"]]); pr = np.array([p["mu"] for p in out["preds"]])
    ss = ((gt - pr) ** 2).sum(); st = ((gt - gt.mean()) ** 2).sum()
    print(f"[check] R²={1-ss/st:+.4f}  MAE={np.abs(gt-pr).mean():.3f}  "
          f"σ mean={np.mean([p['sigma'] for p in out['preds']]):.2f}  N={len(gt)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"[saved] {args.out}  -> feed vào eval_r2_grouped.py / baselines_uq.py")


if __name__ == "__main__":
    main()
