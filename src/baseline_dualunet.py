#!/usr/bin/env python3
import argparse, os, sys, json
import os.path as osp
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.environ.get("DUALUNET_DIR", "DualU-Net"))  # kho ma nguon DualU-Net
from dual_unet.utils.config import load_config
from dual_unet.datasets import build_dataset, build_loader
from dual_unet.models import build_model
from dual_unet.eval.cellsegm_eval_all import find_local_maxima


def find_types_fold3(root, fold=3):
    """Tìm types.npy của fold3 trong dataset PanNuke gốc (bo cu the tuy cach giai nen)."""
    root = Path(root)
    cands = [
        root / f"fold{fold}" / f"Fold {fold}" / "images" / f"fold{fold}" / "types.npy",
        root / f"Fold {fold}" / "images" / f"fold{fold}" / "types.npy",
        root / f"fold{fold}" / "images" / f"fold{fold}" / "types.npy",
    ]
    for c in cands:
        if c.exists():
            return np.load(c, allow_pickle=True)
    hits = list(root.rglob(f"fold{fold}/types.npy")) + list(root.rglob("types.npy"))
    for h in hits:
        if f"fold{fold}" in str(h).lower() or f"fold {fold}" in str(h).lower():
            return np.load(h, allow_pickle=True)
    raise FileNotFoundError(f"Không thấy types.npy fold{fold} dưới {root}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--checkpoint", default="", help="đường dẫn .pth; bỏ trống -> tự lấy .pth MỚI NHẤT trong output_dir của config")
    ap.add_argument("--pannuke_root", default="", help="dataset PanNuke gốc (chứa types.npy) — chỉ cần khi --exclude != ''")
    ap.add_argument("--exclude", default="", help='"" = toan bo fold 3, giu colon (bang chinh); dat ten mo de loai mo do')
    ap.add_argument("--out_csv", default="out/dualunet_fold3_counts.csv")
    args = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config_file)
    cfg["gpu"] = 0
    cfg["distributed"] = False

    do_exclude = bool(args.exclude and args.exclude.strip())
    if do_exclude:
        if not args.pannuke_root:
            sys.exit("--exclude != '' cần --pannuke_root (để lấy types.npy lọc tissue)")
        types3 = find_types_fold3(args.pannuke_root)
        print(f"[types] fold3: {len(types3)} ảnh | ví dụ {types3[:3]}")
    else:
        types3 = None
        print("[types] full fold3 (không loại tissue) — bỏ qua types.npy")

    test_ds = build_dataset(cfg, split="test")
    loader = build_loader(cfg, test_ds, split="test")
    coco = test_ds.coco
    ids = list(test_ds.ids)
    print(f"[data] test fold3: {len(test_ds)} ảnh (COCO)")

    ckpt_path = args.checkpoint
    if not ckpt_path or osp.isdir(ckpt_path):
        ckpt_dir = ckpt_path if ckpt_path else cfg["experiment"]["output_dir"]
        pths = sorted(Path(ckpt_dir).glob("*.pth"), key=lambda p: p.stat().st_mtime)
        if not pths:
            sys.exit(f"Không thấy .pth nào trong {ckpt_dir}")
        ckpt_path = str(pths[-1])
        print(f"[auto] chọn checkpoint mới nhất: {ckpt_path}")
        if len(pths) > 1:
            print(f"[auto] (có {len(pths)} file .pth; lấy mtime lớn nhất)")

    model = build_model(cfg).to(dev)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    th = float(cfg["training"]["th"])
    print(f"[model] loaded {ckpt_path} | h(th)={th}")

    rows = []
    for idx, (images, _target) in enumerate(loader):
        image_id = ids[idx]
        if do_exclude:
            tissue = str(types3[image_id]) if image_id < len(types3) else "?"
            if args.exclude.lower() in tissue.lower():
                continue
        else:
            tissue = "all"
        gt = len(coco.getAnnIds(imgIds=[image_id]))          # số nhân thật
        if isinstance(images, (list, tuple)):
            x = torch.stack([im.to(dev) for im in images])
        else:
            x = images.to(dev)
        out = model(x)
        cmap = out[1][0, 0].detach().cpu().numpy().astype(np.float32)  # (H,W) centroid-map
        _, coords = find_local_maxima(cmap, th)
        pred = int(len(coords))
        rows.append((image_id, tissue, gt, pred))
        if (idx + 1) % 300 == 0:
            print(f"  {idx+1}/{len(test_ds)} ...")

    gt = np.array([r[2] for r in rows], np.float32)
    pr = np.array([r[3] for r in rows], np.float32)
    ss_res = ((gt - pr) ** 2).sum(); ss_tot = ((gt - gt.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = np.abs(gt - pr).mean()

    with open(args.out_csv, "w") as f:
        f.write("image_id,tissue,gt,pred\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")

    print(f"\n=== DualU-Net count (fold_3, loại '{args.exclude}') — leak-free fair baseline ===")
    print(f"  N ảnh (sau loại {args.exclude}): {len(rows)}   (bỏ {len(test_ds)-len(rows)} ảnh {args.exclude})")
    print(f"  GT mean {gt.mean():.1f} | Pred mean {pr.mean():.1f}")
    print(f"  R²  {r2:+.4f}")
    print(f"  MAE {mae:.3f}")
    print(f"  -> CSV/backup: {args.out_csv}")
    print("\nĐiền vào bảng chính cạnh efflite0 (3.6M, count-only, R² 0.908). DualU-Net ~35M params.")


if __name__ == "__main__":
    main()
