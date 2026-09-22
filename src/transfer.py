from __future__ import annotations
import argparse, os, pickle, sys
import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "lib"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from data_io import build_index, find_root  # noqa: E402
from dsunet import (  # noqa: E402
    train, predict_r2, build_pannuke_density, build_nuinsseg_density,
)


def _load_dataset(name, args, device):
    """Trả list mẫu {img, density, gt, organ, [fold]} cho dataset `name` (dùng ĐÚNG cache như bảng chính).

    Bản đồ density luôn dựng từ mặt nạ GT (cache `gt_density_*`). Ở chế độ count-only nó không
    vào hàm mất mát (w_density=0), chỉ có `gt` được dùng."""
    wdir = getattr(args, "work_dir", None) or "work"
    os.makedirs(wdir, exist_ok=True)
    if name == "pannuke":
        folds = [int(x) for x in args.pannuke_folds.split(",")]
        fstr = "".join(str(x) for x in sorted(folds))
        cache = f"{wdir}/gt_density_pannuke_f{fstr}.pkl"
        data = build_pannuke_density(args.pannuke_root, folds, device, cache)
    else:
        cache = f"{wdir}/gt_density_nuinsseg.pkl"
        if os.path.exists(cache):
            samples = None
            print(f"[A/{name}] cache có sẵn -> bỏ qua build_index")
        else:
            nroot = getattr(args, "nuinsseg_root", None) or find_root()
            samples = build_index(nroot)
            print(f"[A/{name}] indexed {len(samples)} pairs (root={nroot})")
        data = build_nuinsseg_density(samples, device, cache)
    # loại mô (colon) nếu bộ dữ liệu này là PanNuke, giống bảng chính
    if args.exclude_tissue and name == "pannuke":
        ex = [t.strip().lower() for t in args.exclude_tissue.split(",") if t.strip()]
        before = len(data)
        data = [d for d in data if not any(e in str(d["organ"]).lower() for e in ex)]
        print(f"[EXCLUDE/{name}] bỏ tissue chứa {ex}: {before} -> {len(data)} ảnh")
    return data


def _load_test_folder(images_dir, gt_csv):
    """Tập kiểm tra từ thư mục ảnh + gt_counts.csv (image,gt_count), cho bất kỳ bộ dữ liệu nào.
    Ảnh được thu nhỏ về 256; organ='_all_' vì ở đây chỉ cần sai số đếm."""
    import csv, glob
    gt = {}
    with open(gt_csv) as f:
        r = csv.reader(f); next(r)
        for row in r:
            gt[row[0]] = float(row[1])
    data = []
    for p in sorted(glob.glob(os.path.join(images_dir, "*.png"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name not in gt:
            continue
        img = np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR)).astype(np.uint8)
        data.append({"img": img, "gt": gt[name], "organ": "_all_"})
    print(f"[TEST-FOLDER] {images_dir}: {len(data)} ảnh (resize 256, OOD zero-shot)")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dataset", choices=["nuinsseg", "pannuke"], required=True)
    ap.add_argument("--test_dataset", choices=["nuinsseg", "pannuke"], default=None,
                    help="dataset test built-in; HOẶC dùng --test_images_dir cho folder ngoài (MoNuSAC).")
    ap.add_argument("--test_images_dir", default=None, help="folder ảnh test ngoài (MoNuSAC) — cần --test_gt_csv")
    ap.add_argument("--test_gt_csv", default=None, help="gt_counts.csv (image,gt_count) cho --test_images_dir")
    ap.add_argument("--pannuke_root", default="data/pannuke")
    ap.add_argument("--pannuke_folds", default="1,2,3")
    ap.add_argument("--exclude_tissue", default=None,
                    help="loại tissue (áp CHO PHÍA PanNuke), vd 'colon'.")
    ap.add_argument("--supervision", choices=["count", "mask"], default="count",
                    help="count = count-only (w_density 0, w_count 1) — chế độ của bài. "
                         "mask = giám sát mật độ dựng từ mặt nạ GT (w_density 1, w_count 0.1). "
                         "Đặt --w_density/--w_count tường minh sẽ ĐÈ lên lựa chọn này.")
    # huan luyen bang dung cau hinh cua bang chinh
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--backbone", default="efficientnet_lite0",
                    help="headline = efficientnet_lite0 (3.62M count-only); tinyunet = model CŨ 1.9M")
    ap.add_argument("--nuinsseg_root", default=None, help="thu muc goc NuInsSeg")
    ap.add_argument("--work_dir", default="work", help="noi luu cache ban do mat do")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--w_density", type=float, default=None, help="None -> suy từ --supervision")
    ap.add_argument("--w_count", type=float, default=None, help="None -> suy từ --supervision")
    ap.add_argument("--w_nll", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--sigma_mode", choices=["poisson", "raw", "nb"], default="poisson")
    ap.add_argument("--detach_mu", action="store_true")
    ap.add_argument("--dump_feat", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="work/xfer_preds.pkl")
    args = ap.parse_args()
    wd_def, wc_def = (0.0, 1.0) if args.supervision == "count" else (1.0, 0.1)
    if args.w_density is None:
        args.w_density = wd_def
    if args.w_count is None:
        args.w_count = wc_def
    print(f"[SUPERVISION] {args.supervision}: w_density={args.w_density} w_count={args.w_count}")

    use_folder = bool(args.test_images_dir)
    assert use_folder or args.test_dataset, "cần --test_dataset HOẶC --test_images_dir + --test_gt_csv"
    assert use_folder or args.train_dataset != args.test_dataset, \
        "cross-dataset: train và test phải KHÁC dataset (in-domain đã có ở bảng chính)."
    if use_folder:
        assert args.test_gt_csv, "--test_images_dir cần kèm --test_gt_csv"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    tgt = args.test_images_dir if use_folder else args.test_dataset
    print(f"device={device} | TRANSFER {args.train_dataset} -> {tgt}")

    # 1) train trên toàn BỘ A (test là dataset khác -> không leak ảnh)
    train_data = _load_dataset(args.train_dataset, args, device)
    print(f"[TRAIN] {args.train_dataset}: {len(train_data)} ảnh (train toàn bộ)")
    model = train(train_data, device, args.epochs, args.ch, args.lr,
                  list(range(len(train_data))), args.w_density, args.w_count, args.w_nll,
                  args.beta, args.bs, args.detach_mu, args.sigma_mode, args.backbone)

    # 2) predict trên toàn BỘ B (built-in dataset hoặc folder ngoài như MoNuSAC)
    if use_folder:
        test_data = _load_test_folder(args.test_images_dir, args.test_gt_csv)
    else:
        test_data = _load_dataset(args.test_dataset, args, device)
    print(f"[TEST] {tgt}: {len(test_data)} ảnh (predict toàn bộ)")
    out = predict_r2(model, test_data, device, dump_feat=args.dump_feat)
    pickle.dump(out, open(args.out, "wb"))

    mu = np.array([p["mu"] for p in out["preds"]])
    sg = np.array([p["sigma"] for p in out["preds"]])
    gt = np.array([g[0] for g in out["gts"]])
    print(f"[OUT] saved {args.out} (N={len(mu)}) | transfer MAE={np.abs(mu-gt).mean():.2f} "
          f"| sigma mean={sg.mean():.2f} std={sg.std():.2f}")
    print(f"  -> chấm conformal (cal trên chính B): "
          f"python eval_r2_grouped.py --preds {args.out} --seeds 20")


if __name__ == "__main__":
    main()
