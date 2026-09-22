from __future__ import annotations
import argparse, csv, os, pickle
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="cache mat do da dung, vd work/gt_density_pannuke_f123.pkl")
    ap.add_argument("--test_fold", type=int, required=True)
    ap.add_argument("--exclude_tissue", default="colon", help="loại mô chứa chuỗi này")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = pickle.load(open(args.cache, "rb"))
    ex = [t.strip().lower() for t in (args.exclude_tissue or "").split(",") if t.strip()]
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows, counts = [], []
    n_excl = 0
    for i, d in enumerate(data):
        if int(d.get("fold", -1)) != args.test_fold:
            continue
        organ = str(d.get("organ", "_all_"))
        if any(e in organ.lower() for e in ex):
            n_excl += 1
            continue
        name = f"f{args.test_fold}_{i:05d}"
        Image.fromarray(d["img"].astype(np.uint8)).save(os.path.join(img_dir, name + ".png"))
        gt = float(d["gt"])
        rows.append([name, gt, organ]); counts.append(gt)

    with open(os.path.join(args.out, "gt_counts.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "gt_count", "organ"]); w.writerows(rows)

    c = np.asarray(counts)
    print(f"XONG -> {img_dir} ({len(rows)} ảnh test fold {args.test_fold}, loại colon {n_excl}) + gt_counts.csv")
    print(f"GT count/ảnh: min={c.min():.0f} max={c.max():.0f} mean={c.mean():.1f} (tổng {c.sum():.0f})")


if __name__ == "__main__":
    main()
