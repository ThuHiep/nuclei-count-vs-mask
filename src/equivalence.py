from __future__ import annotations

import numpy as np


def tost(ec, em, delta=None, delta_frac=0.05, gt=None, alpha=0.05, nboot=10000, seed=0):
    """TOST bắt cặp + bootstrap CI trên hiệu |sai số| từng ảnh.

    ec, em : |sai số| từng ảnh của arm count-only / mask (cùng thứ tự ảnh).
    delta  : biên tương đương, đơn vị NHÂN. None -> delta_frac * số nhân trung bình (cần gt).
    Trả dict; không in gì (in ở `report`).
    """
    ec = np.asarray(ec, float); em = np.asarray(em, float)
    assert ec.shape == em.shape, f"lệch số ảnh: {ec.shape} vs {em.shape}"
    d = ec - em
    n = d.size
    dm = float(d.mean())

    if delta is None:
        assert gt is not None, "cần --gt để suy δ từ delta_frac, hoặc truyền delta trực tiếp"
        delta = float(delta_frac) * float(np.mean(np.asarray(gt, float)))

    # --- TOST tham số (bắt cặp, xấp xỉ chuẩn cho trung bình theo CLT) ---
    from scipy.stats import t as tdist
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    df = n - 1
    if se > 0:
        p_lo = float(tdist.sf((dm + delta) / se, df))   # H0: Δ <= -δ
        p_hi = float(tdist.cdf((dm - delta) / se, df))  # H0: Δ >= +δ
        p_tost = max(p_lo, p_hi)
    else:
        p_lo = p_hi = p_tost = float("nan")

    # --- bootstrap CI (1-2α), không giả định phân phối; |err| lệch phải mạnh ---
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, n, size=(int(nboot), n))].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(bs, [100 * alpha, 100 * (1 - alpha)]))
    delta_star = float(max(abs(lo), abs(hi)))

    return dict(n=n, dm=dm, se=se, delta=float(delta), alpha=float(alpha),
                p_lo=p_lo, p_hi=p_hi, p_tost=p_tost, ci=(lo, hi), delta_star=delta_star,
                mae_c=float(ec.mean()), mae_m=float(em.mean()),
                gt_mean=(float(np.mean(np.asarray(gt, float))) if gt is not None else None),
                equivalent=bool(lo > -delta and hi < delta))


def report(ec, em, gt=None, delta=None, delta_frac=0.05, alpha=0.05, nboot=10000,
           seed=0, label="", indent="  "):
    """Chạy `tost` rồi in gọn. Trả lại dict để gọi ngoài dùng tiếp."""
    r = tost(ec, em, delta=delta, delta_frac=delta_frac, gt=gt, alpha=alpha, nboot=nboot, seed=seed)
    lo, hi = r["ci"]
    pct = (lambda v: f" ({100 * v / r['gt_mean']:.1f}% số nhân TB)") if r["gt_mean"] else (lambda v: "")
    tag = f" [{label}]" if label else ""
    print(f"{indent}--- TƯƠNG ĐƯƠNG (TOST){tag} n={r['n']} ---")
    print(f"{indent}Δ|err| (count − mask) = {r['dm']:+.3f} nhân{pct(abs(r['dm']))}  "
          f"| MAE {r['mae_c']:.2f} vs {r['mae_m']:.2f}")
    print(f"{indent}CI {100 * (1 - 2 * alpha):.0f}% (bootstrap) = [{lo:+.3f}, {hi:+.3f}]  "
          f"| biên δ = ±{r['delta']:.3f}{pct(r['delta'])}")
    print(f"{indent}p_TOST = {r['p_tost']:.3g} (max của hai phía: {r['p_lo']:.3g}, {r['p_hi']:.3g})")
    if r["equivalent"]:
        print(f"{indent}TƯƠNG ĐƯƠNG ở biên ±{r['delta']:.3f} nhân (CI nằm gọn trong biên).")
    else:
        print(f"{indent}CHƯA kết luận được tương đương ở biên ±{r['delta']:.3f}: CI vượt biên.")
    print(f"{indent}δ* = {r['delta_star']:.3f} nhân{pct(r['delta_star'])} "
          f"= biên NHỎ NHẤT còn kết luận được tương đương (báo cáo cái này để khỏi bị cãi là chọn δ cho vừa).")
    return r
