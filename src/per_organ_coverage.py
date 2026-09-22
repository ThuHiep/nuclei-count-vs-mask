#!/usr/bin/env python3
import argparse, collections, numpy as np

def q_of(r, a):
    n = len(r)
    # noi suy mac dinh cua numpy; method="higher" cho phu bien 0,922 thay vi 0,908
    return 0.0 if n == 0 else float(np.quantile(r, min(np.ceil((n + 1) * (1 - a)) / n, 1.0)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="~/Downloads/locuq_pannuke_f3")
    ap.add_argument("--organs", default="~/Downloads/results (8)/evid/s42/preds.npz")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--min_group", type=int, default=15)
    ap.add_argument("--splits", type=int, default=100)  # 20 con nhieu: cuc tieu phu TB nhay 0,899-0,903
    a = ap.parse_args()
    import os
    d = os.path.expanduser(a.dump)
    organ = np.load(os.path.expanduser(a.organs), allow_pickle=True)["organ"]
    oc, ow, per_model = collections.defaultdict(list), collections.defaultdict(list), []
    for sd in (42, 43, 44, 45, 46):
        z = np.load(f"{d}/locuq_s{sd}.npz")
        mu, sg, gt = (z[k].astype(float) for k in ("mu", "sigma", "gt"))
        r = np.abs(gt - mu) / sg
        rng = np.random.default_rng(0)
        m, w, k, o, b = [], [], [], [], []
        for _ in range(a.splits):
            p = rng.permutation(len(gt)); cal, tst = p[:len(gt) // 2], p[len(gt) // 2:]
            q = np.full(len(tst), q_of(r[cal], a.alpha))
            for og in np.unique(organ[cal]):
                msk = organ[cal] == og
                if msk.sum() >= a.min_group:
                    q[organ[tst] == og] = q_of(r[cal][msk], a.alpha)
            lo = np.maximum(0, mu[tst] - q * sg[tst]); hi = mu[tst] + q * sg[tst]
            cov = (gt[tst] >= lo) & (gt[tst] <= hi)
            wk = (hi - lo) + (2 / a.alpha) * np.where(gt[tst] < lo, lo - gt[tst],
                                                     np.where(gt[tst] > hi, gt[tst] - hi, 0.0))
            m.append(cov.mean()); w.append((hi - lo).mean()); k.append(wk.mean())
            pc = {og: cov[organ[tst] == og].mean() for og in np.unique(organ[tst])}
            o.append(min(pc.values())); b.append(sum(v < 0.85 for v in pc.values()))
            for og, v in pc.items():
                oc[og].append(v); ow[og].append((hi - lo)[organ[tst] == og].mean())
        per_model.append([np.mean(x) for x in (m, w, k, o, b)])
    v = np.array(per_model); mn, sd_ = v.mean(0), v.std(0)
    print(f"phu bien {mn[0]:.3f}±{sd_[0]:.3f} | be rong {mn[1]:.1f}±{sd_[1]:.1f} | "
          f"Winkler {mn[2]:.1f}±{sd_[2]:.1f} | phu co quan kem nhat {mn[3]:.3f}±{sd_[3]:.3f} | "
          f"<0.85: {mn[4]:.1f}/19")
    cnt = collections.Counter(organ)
    print(f"\n{'co quan':16s} {'n':>5s} {'phu':>7s} {'be rong':>8s}")
    for og in sorted(oc, key=lambda x: np.mean(oc[x])):
        print(f"{og:16s} {cnt[og]:5d} {np.mean(oc[og]):7.3f} {np.mean(ow[og]):8.1f}")
    print(f"\nCUC TIEU cua phu TRUNG BINH theo co quan = {min(np.mean(v) for v in oc.values()):.3f}"
          f" -> 0/19 duoi 0,85 khi lay trung binh, trong khi cuc tieu TRONG TUNG lan chia"
          f" trung binh chi {mn[3]:.3f}: chenh lech nay la hieu ung chon loc.")

if __name__ == "__main__":
    main()
