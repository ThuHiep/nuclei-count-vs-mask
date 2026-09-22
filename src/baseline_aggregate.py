#!/usr/bin/env python3
import argparse, glob, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True)
ap.add_argument("--name", default="baseline")
a = ap.parse_args()

files = sorted(glob.glob(a.glob))
assert files, f"không thấy file: {a.glob}"
r2s, maes, gt0 = [], [], None
for f in files:
    d = np.loadtxt(f, delimiter=",", skiprows=1)
    gt, pr = d[:, 0], d[:, 1]
    if gt0 is None:
        gt0 = gt
    else:
        assert gt.shape == gt0.shape and np.allclose(gt, gt0), f"gt lệch ở {f} khác tập kiểm tra"
    r2s.append(1 - ((gt - pr) ** 2).sum() / ((gt - gt.mean()) ** 2).sum())
    maes.append(np.abs(gt - pr).mean())
r2s, maes = np.array(r2s), np.array(maes)
print(f"{a.name}  n={len(files)}  R2 {r2s.mean():.3f}±{r2s.std(ddof=1):.3f}  "
      f"MAE {maes.mean():.2f}±{maes.std(ddof=1):.2f}")
print("  per-seed R2:", " ".join(f"{v:.3f}" for v in r2s))
