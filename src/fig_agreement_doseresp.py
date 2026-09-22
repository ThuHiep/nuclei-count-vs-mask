#!/usr/bin/env python3
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]     # slot 1,2,3 — thứ tự cố định, đã kiểm CVD
MARK = ["o", "^", "D"]                         # marker riêng -> phân biệt được cả khi in đen trắng
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


# Nhãn theo ngôn ngữ, mặc định tiếng Anh.
L = {
  "en": dict(gt="Ground-truth count", pred="Predicted count",
             t_a="(a) Agreement with ground truth", t_b="(b) Bland–Altman",
             ba_x="Mean of (predicted, ground truth)", ba_y="Predicted $-$ ground truth",
             bias="bias", loa="95% limits of agreement",
             keep="Fraction of nuclei retained", norm="Predicted count (normalised)",
             ideal="faithful tallying ($y=x$)", rm="nucleus removal",
             sham="sham: nucleus-free tissue removed", sham_s="sham control", dec="."),
  "vi": dict(gt="So nhan thuc", pred="So nhan du doan",
             t_a="(a) Tuong hop du doan-thuc", t_b="(b) Bland–Altman",
             ba_x="Trung binh cua (du doan, thuc)", ba_y="Du doan $-$ thuc",
             bias="do chech", loa="gioi han tuong hop",
             keep="Ti le nhan con lai trong anh", norm="So dem du doan (chuan hoa)",
             ideal="dem cong don trung thuc ($y=x$)", rm="go nhan",
             sham="doi chung gia: go mo khong co nhan", sham_s="doi chung gia", dec=","),
}
LANG = "en"


def num(x, nd=2, sign=False):
    """định dạng số theo ngôn ngữ hiện hành (dấu thập phân . hoặc ,)."""
    t = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return t.replace(".", L[LANG]["dec"])


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
        # RINENG doi anh mau >= 300 dpi. Hinh dinh tinh co raster nhung -> phai dat o day,
        # neu khong pdf xuat ra chi ~226 dpi o kich thuoc in cuoi cung.
        "savefig.dpi": 600,        # Elsevier: combination art (anh + net) >= 500 dpi
    })


def fig_agreement(preds_npz, out):
    z = np.load(preds_npz, allow_pickle=True)
    pred, gt = np.asarray(z["pred"], float), np.asarray(z["gt"], float)
    r2 = 1 - ((gt - pred) ** 2).sum() / ((gt - gt.mean()) ** 2).sum()
    mae = np.abs(pred - gt).mean()
    diff = pred - gt; bias, sd = diff.mean(), diff.std()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.9))
    lim = max(gt.max(), pred.max()) * 1.04
    a1.plot([0, lim], [0, lim], ls="--", lw=1.0, color=INK2, zorder=1)
    a1.scatter(gt, pred, s=4, alpha=0.22, color=SERIES[0], linewidths=0, zorder=2)
    a1.set_xlim(0, lim); a1.set_ylim(0, lim); a1.set_aspect("equal")
    t = L[LANG]
    a1.set_xlabel(t["gt"]); a1.set_ylabel(t["pred"])
    a1.set_title(t["t_a"], loc="left")
    a1.text(0.04, 0.95, f"$R^2$ = {num(r2, 3)}\nMAE = {num(mae)}\n$n$ = {len(gt)}",
            transform=a1.transAxes, va="top", ha="left", color=INK, fontsize=7.5)
    a1.annotate("$y = x$", xy=(lim * 0.72, lim * 0.72), xytext=(lim * 0.82, lim * 0.52),
                color=INK2, fontsize=6.8, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))

    a2.axhline(0, lw=0.6, color=GRID)
    a2.scatter((pred + gt) / 2, diff, s=4, alpha=0.22, color=SERIES[0], linewidths=0)
    for y, ls in [(bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")]:
        a2.axhline(y, ls=ls, lw=1.0, color=SERIES[1])
    xr = a2.get_xlim()[1]
    a2.text(xr * 0.99, bias, f'{t["bias"]} {num(bias, 2, True)}', va="bottom", ha="right", color=INK2, fontsize=6.8)
    a2.text(xr * 0.99, bias + 1.96 * sd, f'{t["loa"]} {num(bias + 1.96 * sd, 1, True)}',
            va="bottom", ha="right", color=INK2, fontsize=6.8)
    a2.text(xr * 0.99, bias - 1.96 * sd, num(bias - 1.96 * sd, 1, True),
            va="top", ha="right", color=INK2, fontsize=6.8)
    a2.set_xlabel(t["ba_x"]); a2.set_ylabel(t["ba_y"])
    a2.set_title(t["t_b"], loc="left")

    fig.tight_layout(w_pad=2.0); fig.savefig(out); plt.close(fig)
    print(f"[hình 1] {out}  (R²={r2:.4f} MAE={mae:.3f} chệch={bias:+.3f} 1.96σ={1.96*sd:.2f})")


def _curve(npz):
    """-> (keep, mean_nhân, sd_nhân, mean_sham). npz có thể là NHIỀU file ngăn bằng dấu phẩy: khi đó các
    phép đo được GỘP, nên dải ±1sd phản ánh cả biến thiên giữa ảnh lẫn giữa các mô hình."""
    fr, pr, fu, sh = [], [], [], []
    for f in str(npz).split(","):
        o = np.load(f.strip(), allow_pickle=True)
        fr.append(o["frac"]); pr.append(o["pred"]); fu.append(o["full"])
        sh.append(o["sham"] if "sham" in o.files else np.full_like(o["pred"], np.nan))
    frac, pred, full = np.concatenate(fr), np.concatenate(pr), np.concatenate(fu)
    sham = np.concatenate(sh)
    ok = full > 1e-6
    norm = np.where(ok, pred / np.where(ok, full, 1), np.nan)
    norm_s = np.where(ok, sham / np.where(ok, full, 1), np.nan)
    keep = np.array(sorted(set(1 - frac)))
    return (keep,
            np.array([np.nanmean(norm[(1 - frac) == k]) for k in keep]),
            np.array([np.nanstd(norm[(1 - frac) == k]) for k in keep]),
            np.array([np.nanmean(norm_s[(1 - frac) == k]) for k in keep]))


def fig_doseresp(occs, out):
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color=INK2, zorder=1)
    # nhãn nằm DỌC theo đường tham chiếu -> khỏi cần mũi tên chỉ, đỡ rối
    t = L[LANG]
    ax.text(0.60, 0.555, t["ideal"], color=INK2, fontsize=6.5,
            rotation=38, rotation_mode="anchor", ha="center", va="top")
    sham_done = False
    for i, (label, npz) in enumerate(occs):
        keep, m, sd, ms = _curve(npz)
        c = SERIES[i % len(SERIES)]
        if i == 0:   # dải ±1sd chỉ cho đường chính; vẽ cho mọi đường thì các dải chồng nhau, không đọc được
            ax.fill_between(keep, m - sd, m + sd, color=c, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(keep, m, "-", marker=MARK[i % len(MARK)], lw=2.0, ms=4.5, color=c, zorder=4,
                label=f'{t["rm"]} ({label})' if len(occs) > 1 else t["rm"])
        if not sham_done and np.isfinite(ms).any():
            ax.plot(keep, ms, "-", marker="s", lw=2.0, ms=4.0, color=SERIES[2], zorder=3,
                    label=t["sham"])
            sham_done = True
        # không ghi nhãn trực tiếp: tên chuỗi tiếng Anh dài, đặt cạnh đường thì đè lên chính đường đó
        # và đè nhãn y=x. Danh tính đã có legend + marker riêng cho từng chuỗi (không chỉ dựa vào màu).
    from matplotlib.ticker import FuncFormatter
    comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", L[LANG]["dec"]))
    ax.xaxis.set_major_formatter(comma); ax.yaxis.set_major_formatter(comma)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 1.12)
    ax.set_xlabel(t["keep"]); ax.set_ylabel(t["norm"])
    ax.legend(loc="lower right", handlelength=1.5, labelspacing=0.3, borderpad=0.2)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    for label, npz in occs:
        keep, m, sd, ms = _curve(npz)
        gain = [(1 - m[j]) / (1 - keep[j]) for j in range(len(keep)) if keep[j] < 1]
        print(f"[hình 2] {label:8s} count khi gỡ hết={m[0]:.3f}  sham={ms[0]:.3f}  "
              f"hệ số đáp ứng={np.mean(gain):.2f}")
    print(f"[hình 2] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="", help="preds.npz -> hình 1")
    ap.add_argument("--occ", action="append", default=[],
                    help='"nhãn=đường/dẫn/occlusion.npz"; lặp lại để so nhiều cách gỡ (cái ĐẦU = chính)')
    ap.add_argument("--out_dir", default=os.environ.get("FIG_DIR", "figures"))
    ap.add_argument("--lang", default="en", choices=["en", "vi"],
                    help="ngôn ngữ nhãn trong hình; đổi luôn dấu thập phân . / ,")
    ap.add_argument("--panel", default="", help="panel_*.png -> hình định tính")
    args = ap.parse_args()
    global LANG
    LANG = args.lang
    os.makedirs(args.out_dir, exist_ok=True)
    style()
    if args.preds:
        fig_agreement(args.preds, os.path.join(args.out_dir, "fig_agreement.pdf"))
    if args.occ:
        occs = [tuple(x.split("=", 1)) for x in args.occ]
        fig_doseresp(occs, os.path.join(args.out_dir, "fig_doseresp.pdf"))
    if args.panel:
        fig_qualitative(args.panel, QUAL[LANG], os.path.join(args.out_dir, "fig_qualitative.pdf"))




def fig_qualitative(panel_png, labels, out):
    """Dựng lại hình định tính từ panel do deletion_doseresp xuất: cắt 4 tile ảnh ra rồi ghi nhãn lại.
    Ly do khong dung thang panel: font matplotlib co the rung dau tieng Viet."""
    from PIL import Image
    a = np.asarray(Image.open(panel_png).convert("RGB"))
    ink = (a.astype(int).sum(2) < 720)                       # pixel không-gần-trắng
    colfull = ink.mean(0) > 0.5                              # cột thuộc vùng ảnh (đặc), không phải chữ
    xs, run = [], None
    for i, v in enumerate(np.r_[colfull, False]):
        if v and run is None: run = i
        elif not v and run is not None:
            if i - run > a.shape[1] // 12: xs.append((run, i))
            run = None
    rows = np.where(ink.mean(1) > 0.5)[0]
    y0, y1 = rows.min(), rows.max()
    assert len(xs) == len(labels), f"cắt được {len(xs)} tile nhưng có {len(labels)} nhãn"
    fig, axes = plt.subplots(1, len(xs), figsize=(6.6, 6.6 / len(xs) + 0.55))
    for ax, (x0, x1), lab in zip(axes, xs, labels):
        ax.imshow(a[y0:y1 + 1, x0:x1]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(GRID); sp.set_linewidth(0.6)
        ax.set_title(lab, fontsize=7, color=INK, linespacing=1.35)
    fig.tight_layout(w_pad=0.8); fig.savefig(out); plt.close(fig)
    print(f"[hình 3] {out}  ({len(xs)} tile, mỗi tile {xs[0][1]-xs[0][0]}×{y1-y0+1} px)")


# Nhãn cho hình định tính. Mảnh dùng ở đây là panel_0063 (29 nhân, hạt giống 42); hệ số đáp ứng
# của riêng mảnh này là 0,81 trong khi toàn tập là 0,62, nên hình chỉ mang tính minh hoạ.
QUAL = {
  "en": ["(a) Original image\n29 annotated nuclei\npredicted 28.7",
         "(b) 30% of nuclei removed\n20 nuclei remain\npredicted 19.9",
         "(c) All nuclei removed\n0 nuclei remain\npredicted 9.1",
         "(d) Sham control\n29 nucleus-free regions, nuclei intact\npredicted 29.6"],
  "vi": ["(a) \u1ea2nh g\u1ed1c\n29 nh\xe2n \u0111\u01b0\u1ee3c ch\xfa th\xedch\nd\u1ef1 \u0111o\xe1n 28,7",
         "(b) G\u1ee1 30% nh\xe2n\nc\xf2n 20 nh\xe2n\nd\u1ef1 \u0111o\xe1n 19,9",
         "(c) G\u1ee1 to\xe0n b\u1ed9 nh\xe2n\nc\xf2n 0 nh\xe2n\nd\u1ef1 \u0111o\xe1n 9,1",
         "(d) \u0110\u1ed1i ch\u1ee9ng gi\u1ea3\n29 v\xf9ng N\u1ec0N, nh\xe2n c\xf2n nguy\xean\nd\u1ef1 \u0111o\xe1n 29,6"],
}


if __name__ == "__main__":
    main()
