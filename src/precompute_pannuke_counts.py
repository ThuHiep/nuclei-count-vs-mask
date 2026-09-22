from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np


def fold_base(root: Path, fold: int) -> Path:
    f = f"fold{fold}"
    for c in (root / f / f"Fold {fold}", root / f"Fold {fold}"):
        if (c / "images" / f / "images.npy").exists():
            return c
    raise FileNotFoundError(f"Không thấy Fold {fold} dưới {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", required=True)
    ap.add_argument("--folds", default="1,2,3")
    args = ap.parse_args()
    root = Path(args.pannuke_root)
    for fold in [int(x) for x in args.folds.split(",")]:
        base = fold_base(root, fold)
        f = f"fold{fold}"
        masks_path = base / "masks" / f / "masks.npy"
        out_path = base / "images" / f / "counts.npy"
        if not masks_path.exists():
            print(f"[fold {fold}] masks.npy KHÔNG còn ({masks_path}) — bỏ qua "
                  f"(counts.npy {'đã có' if out_path.exists() else 'CHƯA có -> lỗi!'})")
            continue
        masks = np.load(masks_path, mmap_mode="r")   # (N,256,256,6)
        n = masks.shape[0]
        counts = np.zeros((n, 5), np.int32)
        t0 = time.time()
        for i in range(n):
            m = np.asarray(masks[i, :, :, :5], dtype=np.int32)   # (256,256,5)
            for k in range(5):
                counts[i, k] = int(np.unique(m[:, :, k]).size - 1)
            if (i + 1) % 500 == 0:
                print(f"[fold {fold}] {i+1}/{n} {(time.time()-t0)/(i+1):.3f}s/img")
        np.save(out_path, counts)
        print(f"[fold {fold}] saved {out_path} shape={counts.shape} "
              f"total_nuclei={int(counts.sum())} | giờ có thể xoá {masks_path}")


if __name__ == "__main__":
    main()
