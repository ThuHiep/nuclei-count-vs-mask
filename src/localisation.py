#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "lib"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from dsunet import build_pannuke_density, train
from losses import count_from_density


@torch.no_grad()
def localize_stats(model, test_data, dev):
    """-> dict(corr, mass, chance, ratio, r2). ratio=mass/chance (>1 = localize hơn ngẫu nhiên)."""
    model.eval()
    corrs, massfr, chance, pr, yt = [], [], [], [], []
    for d in test_data:
        x = torch.from_numpy(d["img"].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        dens = model(x)[0]                                  # (1,1,H,W)
        pred = dens.squeeze().cpu().numpy().astype(np.float64)   # (H,W)
        gt = d["density"].astype(np.float64)                # (H,W) GT nhân
        pr.append(float(count_from_density(dens)[0])); yt.append(d["gt"])
        nuc = gt > 0
        chance.append(float(nuc.mean()))                    # tỉ lệ pixel là nhân (= mass kỳ vọng nếu density đều)
        pf, gf = pred.ravel(), gt.ravel()
        if pf.std() > 1e-8 and gf.std() > 1e-8:
            corrs.append(float(np.corrcoef(pf, gf)[0, 1]))
        tot = pred.sum()
        if abs(tot) > 1e-8:
            massfr.append(float(pred[nuc].sum() / tot))     # khối rơi lên pixel nhân GT
    pr, yt = np.array(pr), np.array(yt)
    ss = ((yt - pr) ** 2).sum(); st = ((yt - yt.mean()) ** 2).sum()
    r2 = 1 - ss / st if st > 0 else float("nan")
    corr = float(np.mean(corrs)) if corrs else float("nan")
    mass = float(np.mean(massfr)) if massfr else float("nan")
    ch = float(np.mean(chance))
    return {"corr": corr, "mass": mass, "chance": ch,
            "ratio": (mass / ch if ch > 0 and mass == mass else float("nan")), "r2": r2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", required=True)
    ap.add_argument("--train_folds", default="1,2")
    ap.add_argument("--test_fold", default="3")
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--w_count_mask", type=float, default=0.1,
                    help="neo count cho baseline mask (0.01 SẬP density-MSE; 0.1 để mask hội tụ, so localize công bằng)")
    ap.add_argument("--dump_npz", default=None,
                    help="lưu (img,gt,pred_count,pred_mask) của N ảnh test -> vẽ Figure localize (lọc đẹp sau)")
    ap.add_argument("--n_dump", type=int, default=40, help="số ảnh ứng viên (rải đều test, đa dạng mô) để lọc")
    ap.add_argument("--work_dir", default="work")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:                                       # work_dir co the chi doc
        os.makedirs(args.work_dir, exist_ok=True)
    except OSError as e:
        print(f"[work_dir] không tạo được ({e}) — chỉ đọc cache")

    tr_folds = [int(x) for x in args.train_folds.split(",")]
    fstr = "".join(str(x) for x in sorted(tr_folds))
    train_data = build_pannuke_density(args.pannuke_root, tr_folds, dev,
                                       f"{args.work_dir}/gt_density_pannuke_f{fstr}.pkl")
    test_data = build_pannuke_density(args.pannuke_root, [int(args.test_fold)], dev,
                                      f"{args.work_dir}/gt_density_pannuke_f{args.test_fold}.pkl")
    tr = list(range(len(train_data)))
    print(f"[data] train fold{tr_folds}={len(train_data)} / test fold{args.test_fold}={len(test_data)} (GT density)")

    seeds = [int(s) for s in args.seeds.split(",")]
    configs = [("count-only", 0.0, 1.0), ("mask (GT-density)", 1.0, args.w_count_mask)]
    agg = {name: {"corr": [], "mass": [], "ratio": [], "r2": []} for name, _, _ in configs}
    dump_models = {}
    for name, w_d, w_c in configs:
        for seed in seeds:
            np.random.seed(seed); torch.manual_seed(seed)
            model = train(train_data, dev, args.epochs, args.ch, 1e-3, tr,
                          w_d, w_c, 0.01, 0.5, 16, True, "poisson", args.backbone)
            st = localize_stats(model, test_data, dev)
            for k in agg[name]:
                agg[name][k].append(st[k])
            print(f"  [{name:18s} seed{seed}] corr {st['corr']:+.4f} | mass {st['mass']:.3f} "
                  f"(ratio {st['ratio']:.2f}×) | count-R² {st['r2']:+.4f}")
            if args.dump_npz and seed == seeds[0]:
                dump_models[name] = model

    chance = float(np.mean([ (d["density"] > 0).mean() for d in test_data ]))
    print(f"\n=== LOCALIZATION EMERGENCE (fold3 test, efflite0, {len(seeds)} seed) ===")
    print(f"  chance (nhân chiếm) = {chance:.3f} | GT hoàn hảo ratio = {1/chance:.2f}× | uniform = 1.0×")
    print(f"  {'supervision':18s} {'ratio(mass conc.)':>22s} {'corr':>10s} {'count-R²':>12s}")
    for name, _, _ in configs:
        a = agg[name]
        print(f"  {name:18s}  {np.mean(a['ratio']):5.2f}× ± {np.std(a['ratio']):.2f}"
              f"        {np.mean(a['corr']):+.4f}   {np.mean(a['r2']):+.4f} ± {np.std(a['r2']):.4f}")
    print("\n  ratio lớn hơn 1 nghĩa là mật độ tập trung lên nhân.")
    print("       = spatial localization TỰ NỔI từ count, SẮC hơn density-MSE (headline cơ chế).")

    if args.dump_npz and len(dump_models) == 2:
        import torch as _t
        # rải đều khắp test-set để đa dạng mô (thay vì N ảnh đầu cùng 1 tissue)
        idxs = np.linspace(0, len(test_data) - 1, min(args.n_dump, len(test_data))).astype(int)
        ex = {"chance": chance, "idxs": idxs}
        for n, j in enumerate(idxs):
            d = test_data[int(j)]
            ex[f"img_{n}"] = d["img"]; ex[f"gt_{n}"] = d["density"].astype(np.float32)
            ex[f"organ_{n}"] = str(d.get("organ", "")); ex[f"gtcount_{n}"] = float(d["gt"])
            x = _t.from_numpy(d["img"].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
            with _t.no_grad():
                ex[f"count_{n}"] = dump_models["count-only"](x)[0].squeeze().cpu().numpy().astype(np.float32)
                ex[f"mask_{n}"] = dump_models["mask (GT-density)"](x)[0].squeeze().cpu().numpy().astype(np.float32)
        np.savez_compressed(args.dump_npz, **ex)
        print(f"\n  [dump] {len(idxs)} ảnh ứng viên (rải đều, kèm organ) -> {args.dump_npz} — tôi lọc đẹp nhất vẽ Figure")


if __name__ == "__main__":
    main()
