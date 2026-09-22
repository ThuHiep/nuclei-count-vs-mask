#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def _fold_base(root, fold):
    f = f"fold{fold}"
    for c in (root / f / f"Fold {fold}", root / f"Fold {fold}"):
        if (c / "images" / f / "images.npy").exists():
            return c
    raise FileNotFoundError(f"Không thấy Fold {fold} dưới {root}")


def merge_instance(mask5):
    """mask5 (256,256,5) instance-labeled mỗi kênh -> 1 label-map, ID toàn cục duy nhất."""
    inst = np.zeros(mask5.shape[:2], np.int32)
    cur = 0
    for k in range(5):
        lab = mask5[:, :, k].astype(np.int32)
        m = lab > 0
        if m.any():
            inst[m] = lab[m] + cur
            cur = int(inst.max())
    return inst


def load_fold(root, fold, exclude, want_label):
    base = _fold_base(root, fold)
    d = base / "images" / f"fold{fold}"
    imgs = np.load(d / "images.npy", mmap_mode="r")
    tp = d / "types.npy"
    types = np.load(tp, allow_pickle=True) if tp.exists() else np.array(["na"] * len(imgs))
    masks = np.load(base / "masks" / f"fold{fold}" / "masks.npy", mmap_mode="r")
    X, Y, C = [], [], []
    n_colon, n_drop = 0, 0
    for i in range(len(imgs)):
        t = str(types[i])
        if "colon" in t.lower():
            n_colon += 1
        if exclude and exclude.lower() in t.lower():
            n_drop += 1
            continue
        im = np.asarray(imgs[i])
        if im.max() <= 1.5:
            im = im * 255
        X.append(im.astype(np.uint8))
        inst = merge_instance(np.asarray(masks[i, :, :, :5], np.int32))
        C.append(int(len(np.unique(inst)) - (1 if (inst == 0).any() else 0)))  # GT count
        if want_label:
            Y.append(inst)
        if (i + 1) % 800 == 0:
            print(f"  [fold{fold}] {i+1}/{len(imgs)}")
    kept = len(imgs) - n_drop
    print(f"  [fold{fold}] tổng {len(imgs)} | giữ {kept} | bỏ {n_drop} (exclude='{exclude or 'NONE'}') "
          f"| colon trong fold: {n_colon} -> {'GIỮ colon ' if n_drop == 0 else 'ĐÃ LOẠI'}")
    return X, (Y if want_label else None), np.array(C, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", required=True)
    ap.add_argument("--train_folds", default="1,2")
    ap.add_argument("--test_fold", default="3")
    ap.add_argument("--exclude_tissue", default="")   # "" = full fold3 (khớp efflite0 0.920)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", default="")   # rong -> tu dat theo seed
    args = ap.parse_args()
    args.out_csv = args.out_csv or f"out/stardist_fold3_counts_s{args.seed}.csv"
    # kiem quyen ghi truoc khi train
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    _p = Path(args.out_csv + ".probe"); _p.write_text(""); _p.unlink()   # thu ghi roi xoa, khong tao san out_csv

    from csbdeep.utils import normalize
    from stardist import fill_label_holes, calculate_extents
    from stardist.models import Config2D, StarDist2D
    import random
    import tensorflow as tf
    # seed init + thu tu batch + split val. cuDNN van khong bit-exact -> tai lap XAP XI, khong tuyet doi.
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    print(f"[tf] {tf.__version__} | GPU {tf.config.list_physical_devices('GPU')} | seed {args.seed}")

    root = Path(args.pannuke_root)
    trX, trY = [], []
    for f in [int(x) for x in args.train_folds.split(",")]:
        print(f"[load] train fold{f} ...")
        X, Y, _ = load_fold(root, f, args.exclude_tissue, want_label=True)
        trX += X; trY += Y
    print(f"[load] test fold{args.test_fold} ...")
    teX, _, teC = load_fold(root, int(args.test_fold), args.exclude_tissue, want_label=False)
    print(f"[data] train {len(trX)} / test {len(teX)} | GT count mean {teC.mean():.1f}")

    # chuẩn hoá + fill holes
    axis_norm = (0, 1)
    trX = [normalize(x, 1, 99.8, axis=axis_norm) for x in trX]
    trY = [fill_label_holes(y) for y in trY]
    teXn = [normalize(x, 1, 99.8, axis=axis_norm) for x in teX]

    # split nhỏ val cho StarDist (10%)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(trX)); nval = max(1, len(trX) // 10)
    vi, ti = idx[:nval], idx[nval:]
    Xt = [trX[i] for i in ti]; Yt = [trY[i] for i in ti]
    Xv = [trX[i] for i in vi]; Yv = [trY[i] for i in vi]

    n_rays = 32
    grid = (2, 2)
    conf = Config2D(n_rays=n_rays, grid=grid, use_gpu=False, n_channel_in=3,   # use_gpu=gputools/OpenCL; tắt -> CPU cho NMS, mạng vẫn GPU qua TF
                    train_epochs=args.epochs, train_steps_per_epoch=args.steps,
                    train_patch_size=(256, 256), train_batch_size=8)
    model = StarDist2D(conf, name=f"stardist_pannuke_f12_s{args.seed}", basedir="work/stardist_models")
    med = calculate_extents(list(Yt), np.median)
    print(f"[stardist] median object extent {med}")

    model.train(Xt, Yt, validation_data=(Xv, Yv), augmenter=None)
    model.optimize_thresholds(Xv, Yv)

    # đếm fold3 = số instance dự đoán
    pred = []
    for j, x in enumerate(teXn):
        labels, _ = model.predict_instances(x)
        pred.append(int(len(np.unique(labels)) - (1 if (labels == 0).any() else 0)))
        if (j + 1) % 300 == 0:
            print(f"  [pred] {j+1}/{len(teXn)}")
    pred = np.array(pred, np.float32)

    ss_res = ((teC - pred) ** 2).sum(); ss_tot = ((teC - teC.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = np.abs(teC - pred).mean()

    with open(args.out_csv, "w") as f:
        f.write("gt,pred\n")
        for g, p in zip(teC, pred):
            f.write(f"{int(g)},{int(p)}\n")

    print(f"\n=== StarDist count (fold_3, loại '{args.exclude_tissue or 'none'}', seed {args.seed}) — leak-free ===")
    print(f"  N ảnh {len(teX)} | GT mean {teC.mean():.1f} | Pred mean {pred.mean():.1f}")
    print(f"  R²  {r2:+.4f}")
    print(f"  MAE {mae:.3f}")
    print(f"  -> CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
