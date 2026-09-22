from __future__ import annotations
import argparse, csv, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_io import build_index, find_root, _load_mask  # noqa: E402


def gt_count_from_mask(m: np.ndarray) -> int:
    """Số nhân = số nhãn khác nhau trong mặt na, trừ nền 0."""
    return int(len(np.unique(m)) - (1 if (m == 0).any() else 0))


def save_resize(img: np.ndarray, size: int):
    return [(Image.fromarray(img).resize((size, size), Image.BILINEAR), "")]  # 1 ảnh, hậu tố rỗng


def save_tiles(img: np.ndarray, size: int):
    """Cắt lưới không chồng, pad phản chiếu tới bội `size`. Trả list (PIL, suffix _rIcJ)."""
    H, W = img.shape[:2]
    ph, pw = (-H) % size, (-W) % size
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    Hh, Ww = img.shape[:2]
    out = []
    for i in range(0, Hh, size):
        for j in range(0, Ww, size):
            out.append((Image.fromarray(img[i:i+size, j:j+size]), f"_r{i//size}c{j//size}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="NuInsSeg root (mặc định find_root())")
    ap.add_argument("--out", required=True, help="thư mục xuất")
    ap.add_argument("--mode", choices=["resize", "tile"], default="resize")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    root = args.root or find_root()
    samples = build_index(root)
    print(f"NuInsSeg root={root}  #images={len(samples)}  mode={args.mode} size={args.size}")

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    gt_rows, type_rows, tile_rows = [], [], []
    n_nuc = 0
    for k, s in enumerate(samples):
        organ = s["organ"]
        img = np.asarray(Image.open(s["image"]).convert("RGB"))
        m = _load_mask(s["mask"])
        gt = gt_count_from_mask(m)
        n_nuc += gt
        # tên duy nhất: organ + stem (stem có thể trùng giữa các organ)
        stem = f"{organ}__{os.path.splitext(os.path.basename(s['image']))[0]}".replace(" ", "_")
        pieces = save_resize(img, args.size) if args.mode == "resize" else save_tiles(img, args.size)
        for pil, suf in pieces:
            name = f"{stem}{suf}"
            pil.save(os.path.join(img_dir, name + ".png"))
            type_rows.append([name, organ])
            if args.mode == "tile":
                tile_rows.append([name, stem])   # tile -> ảnh gốc (để cộng count)
        gt_rows.append([stem, gt, organ])         # GT count theo ảnh GỐC (full-res)
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(samples)} imgs, {n_nuc} nuclei")

    with open(os.path.join(args.out, "gt_counts.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["image", "gt", "organ"], *gt_rows])
    with open(os.path.join(args.out, "types.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["image", "tissue"], *type_rows])
    if args.mode == "tile":
        with open(os.path.join(args.out, "tiles_map.csv"), "w", newline="") as f:
            csv.writer(f).writerows([["tile", "image"], *tile_rows])
    print(f"XONG -> {args.out}  (gt_counts.csv {len(gt_rows)} ảnh, {n_nuc} nhân, images/ {len(type_rows)} file)")
    print("Bước tiếp: chạy CellViT/LKCell inference trên images/, dump image->len(instance_types) ra preds.csv,")
    print("rồi: python baseline_heavy_eval.py --gt <out>/gt_counts.csv --preds preds.csv [--tiles_map <out>/tiles_map.csv]")


if __name__ == "__main__":
    main()
