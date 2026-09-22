#!/usr/bin/env python3
import argparse, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "lib"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
from data_io import build_index, find_root
from dsunet import build_nuinsseg_density, train
from losses import count_from_density


@torch.no_grad()
def per_image_pred(model, data, idx, dev):
    model.eval(); pr = []
    for i in idx:
        x = torch.from_numpy(data[i]["img"].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        pr.append(float(count_from_density(model(x)[0])[0]))
    return np.array(pr)


def r2mae(pr, yt):
    ss = ((yt - pr) ** 2).sum(); st = ((yt - yt.mean()) ** 2).sum()
    return (1 - ss / st if st > 0 else float("nan")), float(np.abs(pr - yt).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuinsseg_root", default=None)
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", default="42,43,44,45,46", help="mỗi seed 1 split -> per-image Wilcoxon; xem nhất quán")
    ap.add_argument("--w_count_mask", type=float, default=0.1)
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--delta", type=float, default=None,
                    help="TOST: biên tương đương (đơn vị NHÂN). Mặc định None -> suy từ --delta_frac.")
    ap.add_argument("--delta_frac", type=float, default=0.05,
                    help="TOST: biên = delta_frac * số nhân trung bình của tập kiểm tra (mặc định 5%%).")
    ap.add_argument("--dump_preds", default=None,
                    help="lưu .npz (seeds, ec, em, gt theo từng phép chia) để tính lại TOST mà không phải huấn luyện lại. "
                         "nên đặt ngoài kho mã nguồn.")
    ap.add_argument("--deterministic", action="store_true",
                    help="bật cuDNN tất định — CHẬM hơn nhiều với efflite0. NuInsSeg nhỏ+ổn định KHÔNG cần "
                         "-> mặc định TẮT (benchmark=True) cho nhanh.")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:                                       # work_dir co the chi doc
        os.makedirs(args.work_dir, exist_ok=True)
    except OSError as e:
        print(f"[work_dir] không tạo được ({e}) — chỉ đọc cache")
    from scipy.stats import wilcoxon

    cache = f"{args.work_dir}/gt_density_nuinsseg.pkl"
    if os.path.exists(cache):        # có cache thì không cần dữ liệu gốc
        samples = None
        print(f"[data] cache có sẵn ({cache}) -> bỏ qua build_index")
    else:
        samples = build_index(args.nuinsseg_root or find_root())
    data = build_nuinsseg_density(samples, dev, cache)
    n = len(data)
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"[data] NuInsSeg {n} ảnh (GT density từ mask) | {len(seeds)} seed")

    r2c_all, r2m_all, rows, err_all = [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed); idx = rng.permutation(n); n_te = n // 5
        te, tr = idx[:n_te], idx[n_te:]
        gt = np.array([data[i]["gt"] for i in te])
        preds = {}
        for name, wd, wc in [("count-only", 0.0, 1.0), ("mask (GT-density)", 1.0, args.w_count_mask)]:
            np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            if args.deterministic:
                torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True, warn_only=True)
            else:
                torch.backends.cudnn.benchmark = True   # nhanh; NuInsSeg nhỏ+ổn định nên không cần tất định
            model = train(data, dev, args.epochs, args.ch, 1e-3, list(tr),
                          wd, wc, 0.01, 0.5, 16, True, "poisson", args.backbone)
            preds[name] = per_image_pred(model, data, te, dev)
        ec = np.abs(gt - preds["count-only"]); em = np.abs(gt - preds["mask (GT-density)"])
        r2c, _ = r2mae(preds["count-only"], gt); r2m, _ = r2mae(preds["mask (GT-density)"], gt)
        try:
            _, p = wilcoxon(ec, em)
        except Exception as e:
            p = float("nan"); print(f"[wilcoxon lỗi seed{seed}] {e}")
        dmean = float((ec - em).mean())
        r2c_all.append(r2c); r2m_all.append(r2m); rows.append((seed, r2c, r2m, ec.mean(), em.mean(), dmean, p))
        err_all.append((seed, ec, em, gt))
        vd = "parity" if p > 0.05 else ("count>mask" if dmean < 0 else "count<mask")
        print(f"  [seed{seed}] R² count {r2c:+.4f} / mask {r2m:+.4f} | |err| {ec.mean():.2f} vs {em.mean():.2f} "
              f"| p={p:.4g} -> {vd}")

    n_sig_win = sum(1 for r in rows if r[6] < 0.05 and r[5] < 0)
    n_win_dir = sum(1 for r in rows if r[5] < 0)
    print(f"\n=== NuInsSeg PARITY + per-image Wilcoxon (efflite0, {len(seeds)} seed) ===")
    print(f"  count-only R² {np.mean(r2c_all):+.4f} ± {np.std(r2c_all):.4f} | "
          f"mask R² {np.mean(r2m_all):+.4f} ± {np.std(r2m_all):.4f}")
    print(f"  count-only lỗi ÍT hơn ở {n_win_dir}/{len(seeds)} seed | p<0.05 ở {n_sig_win}/{len(seeds)} seed")
    print(f"  -> {'count-only lỗi ít hơn ở MỌI seed' if n_win_dir == len(seeds) else f'chiều lỗi đổi dấu giữa các seed ({n_win_dir}/{len(seeds)})'}"
          f"{'; có seed khác biệt CÓ ý nghĩa -> KHÔNG phải parity thuần' if n_sig_win else '; không seed nào khác biệt có ý nghĩa -> nhất quán với TƯƠNG ĐƯƠNG'}")
    if args.dump_preds:   # lưu trước khi chạy TOST: TOST lỗi cũng không mất dữ liệu train
        os.makedirs(os.path.dirname(args.dump_preds) or ".", exist_ok=True)
        np.savez(args.dump_preds,
                 seeds=np.array([s for s, _, _, _ in err_all]),
                 ec=np.stack([e for _, e, _, _ in err_all]),
                 em=np.stack([m for _, _, m, _ in err_all]),
                 gt=np.stack([g for _, _, _, g in err_all]))
        print(f"\n[dump_preds] đã lưu {args.dump_preds}")

    # TOST theo từng seed: mỗi seed là một split riêng (test set khác nhau) nên không gộp per-image
    # được; gộp sẽ lặp ảnh giữa các split -> vi phạm độc lập. n≈26/split nên lực kiểm định thấp,
    # δ* sẽ rộng: đó là sự thật về dữ liệu, không phải lỗi của test.
    try:
        from equivalence import tost
        print(f"\n  --- TƯƠNG ĐƯƠNG (TOST) theo từng split ---")
        eq, ds = 0, []
        for seed, ec, em, gt in err_all:
            r = tost(ec, em, gt=gt, delta=args.delta, delta_frac=args.delta_frac)
            eq += bool(r["equivalent"]); ds.append(r["delta_star"])
            print(f"  [seed{seed}] Δ={r['dm']:+.3f} CI90 [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] "
                  f"δ=±{r['delta']:.3f} p_TOST={r['p_tost']:.3g} δ*={r['delta_star']:.3f}"
                  f" -> {'TƯƠNG ĐƯƠNG' if r['equivalent'] else 'chưa kết luận'}")
        print(f"  tương đương ở {eq}/{len(err_all)} split (biên {args.delta_frac:.0%} số nhân TB); "
              f"δ* trung vị = {float(np.median(ds)):.3f} nhân")
    except Exception as e:
        print(f"  [TOST lỗi] {e}")


if __name__ == "__main__":
    main()
