#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "lib"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from data_io import build_index, find_root, _load_mask, IMG_SIZE


def to_label(m):
    m = np.asarray(m)
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(np.int32)


def count_inst(m):
    u = np.unique(m)
    return int(len(u) - (1 if (u == 0).any() else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuinsseg_root", default=None)
    ap.add_argument("--seed", type=int, default=42, help="hạt giống cho phép chia 80/20")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out_csv", default="out/stardist_nuinsseg_counts.csv")
    args = ap.parse_args()

    root = args.nuinsseg_root or find_root()
    samples = build_index(root)
    print(f"[data] NuInsSeg {len(samples)} ảnh (root={root})")

    imgs, labels, gts = [], [], []
    for s in samples:
        im = np.asarray(Image.open(s["image"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        m0 = to_label(_load_mask(s["mask"]))
        gts.append(count_inst(m0))                                   # GT count trên mask GỐC
        lab = np.asarray(Image.fromarray(m0, mode="I").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)).astype(np.int32)
        imgs.append(im); labels.append(lab)
    n = len(imgs); gts = np.array(gts, np.float32)

    # phép chia giống parity_nuinsseg: rng(seed), n_te = n//5, te = idx[:n_te]
    rng = np.random.default_rng(args.seed); idx = rng.permutation(n); n_te = n // 5
    te, tr = idx[:n_te], idx[n_te:]
    teC = gts[te]
    print(f"[split] seed{args.seed}: train {len(tr)} / test {len(te)} | GT count mean {teC.mean():.1f}")

    from csbdeep.utils import normalize
    from stardist import fill_label_holes, calculate_extents
    from stardist.models import Config2D, StarDist2D

    trX = [normalize(imgs[i], 1, 99.8, axis=(0, 1)) for i in tr]
    trY = [fill_label_holes(labels[i]) for i in tr]
    teXn = [normalize(imgs[i], 1, 99.8, axis=(0, 1)) for i in te]

    rng2 = np.random.default_rng(42); ii = rng2.permutation(len(trX)); nval = max(1, len(trX) // 10)
    vi, ti = ii[:nval], ii[nval:]
    Xt = [trX[i] for i in ti]; Yt = [trY[i] for i in ti]
    Xv = [trX[i] for i in vi]; Yv = [trY[i] for i in vi]

    conf = Config2D(n_rays=32, grid=(2, 2), use_gpu=False, n_channel_in=3,
                    train_epochs=args.epochs, train_steps_per_epoch=args.steps,
                    train_patch_size=(256, 256), train_batch_size=8)
    model = StarDist2D(conf, name="stardist_nuinsseg", basedir="work/stardist_models")
    print(f"[stardist] median object extent {calculate_extents(list(Yt), np.median)}")
    model.train(Xt, Yt, validation_data=(Xv, Yv), augmenter=None)
    model.optimize_thresholds(Xv, Yv)

    pred = []
    for j, x in enumerate(teXn):
        lab_p, _ = model.predict_instances(x)
        pred.append(count_inst(lab_p))
        if (j + 1) % 100 == 0:
            print(f"  [pred] {j+1}/{len(teXn)}")
    pred = np.array(pred, np.float32)

    ss = ((teC - pred) ** 2).sum(); st = ((teC - teC.mean()) ** 2).sum()
    r2 = 1 - ss / st if st > 0 else float("nan"); mae = np.abs(teC - pred).mean()
    with open(args.out_csv, "w") as f:
        f.write("gt,pred\n")
        for g, p in zip(teC, pred):
            f.write(f"{int(g)},{int(p)}\n")

    print(f"\n=== StarDist trong miền, NuInsSeg (hạt giống {args.seed}) ===")
    print(f"  N {len(te)} | GT mean {teC.mean():.1f} | Pred mean {pred.mean():.1f}")
    print(f"  R²  {r2:+.4f}")
    print(f"  MAE {mae:.3f}")
    print(f"  -> so efflite0 count-only NuInsSeg 0.854 (3-seed). CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
