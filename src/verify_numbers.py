import os
import itertools
import pickle

import numpy as np

D = os.environ.get("DATA_DIR", "data")
S = [42, 43, 44, 45, 46]
B, rng = 10_000, np.random.default_rng(0)

r2 = lambda g, p: 1 - ((g - p) ** 2).sum() / ((g - g.mean()) ** 2).sum()


def dstar(d):                                     # max(|L|,|U|) cua CI bootstrap 90%
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    L, U = np.percentile(bs, [5, 95])
    return max(abs(L), abs(U))


# ---- A. PanNuke fold3, 5 hat giong moi nhanh -------------------------------
gt = np.load(f"{D}/results (8)/evid/s42/preds.npz")["gt"]
cnt = np.stack([np.load(f"{D}/results (8)/evid/s{s}/preds.npz")["pred"] for s in S])
mk = np.stack([np.load(f"{D}/pannuke_mask_5seed/outputs/maskbal_s{s}.npz")["pred__GT-density_(mask)"][0]
               for s in S])
m = gt.mean()
print(f"[A] PanNuke n={len(gt)} so nhan TB={m:.4f}")
for nm, P in (("count", cnt), ("mask ", mk)):
    print(f"  {nm} R2 " + " ".join(f"s{s}:{r2(gt, p):.3f}" for s, p in zip(S, P)) +
          f"  | mean {np.mean([r2(gt, p) for p in P]):.4f}±{np.std([r2(gt, p) for p in P]):.4f}"
          f"  | MAE " + " ".join(f"{np.abs(gt - p).mean():.2f}" for p in P))
d = np.abs(gt - cnt).mean(0) - np.abs(gt - mk).mean(0)          # bat cap: TB sai so theo seed truoc
ds = dstar(d)
print(f"  giua hai che do: Dhat={d.mean():+.4f} ({100 * abs(d.mean()) / m:.2f}%)  "
      f"dstar={ds:.3f} nhan = {100 * ds / m:.2f}%")
for nm, P in (("count", cnt), ("mask ", mk)):
    v = np.array([100 * dstar(np.abs(gt - P[a]) - np.abs(gt - P[b])) / m
                  for a, b in itertools.combinations(range(5), 2)])
    print(f"  {nm} doi seed 10 cap: trung vi {np.median(v):.2f}%  dai {v.min():.2f}-{v.max():.2f}%  "
          + " ".join(f"{S[a]}-{S[b]}:{x:.2f}" for (a, b), x in zip(itertools.combinations(range(5), 2), v)))

# ---- B. Chuyen mien PanNuke -> NuInsSeg, he so thang do --------------------
XF = [("count", s, f"{D}/pannuke-nuinsseg_count/xfer_count_s{s}.pkl") for s in (42, 43, 44)] + \
     [("mask ", s, f"{D}/results-12/outputs/xfer_mask_bal_s{s}.pkl") for s in (42, 43, 44)]
print("\n[B] chuyen mien: arm seed | tho | k+MAE+R2 in-sample | k+MAE+R2 nua de rieng (200 lan chia)")
for arm, s, f in XF:
    o = pickle.load(open(f, "rb"))
    p = np.array([x["mu"] for x in o["preds"]])
    g = np.array([float(x[0]) for x in o["gts"]])
    if (p * p).sum() < 1e-9:
        print(f"  {arm} {s} suy bien (sigma=0, R2={r2(g, p):+.3f}) -> khong tinh he so")
        continue
    k = (p * g).sum() / (p * p).sum()
    ks, ms, rs = [], [], []
    for _ in range(200):
        i = rng.permutation(len(g)); c, t = i[:len(g) // 2], i[len(g) // 2:]
        kc = (p[c] * g[c]).sum() / (p[c] * p[c]).sum()
        ks.append(kc); ms.append(np.abs(g[t] - kc * p[t]).mean()); rs.append(r2(g[t], kc * p[t]))
    print(f"  {arm} {s} | {np.abs(g - p).mean():5.2f} {r2(g, p):+.3f} | {k:.2f} {np.abs(g - k * p).mean():5.2f} "
          f"{r2(g, k * p):+.3f} | {np.mean(ks):.2f}±{np.std(ks):.3f} {np.mean(ms):5.2f} {np.mean(rs):+.3f}")
