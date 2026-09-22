#!/usr/bin/env python3
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, ACCENT = "#1b1b1b", "#8a8a8a", "#2f6f9f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm", type=float, default=-0.3068, help="Δ|err| trung bình (nhân)")
    ap.add_argument("--lo", type=float, default=-0.3578, help="cận dưới CI 90%%")
    ap.add_argument("--hi", type=float, default=-0.2562, help="cận trên CI 90%%")
    ap.add_argument("--gt_mean", type=float, default=24.4871, help="số nhân trung bình/ảnh")
    ap.add_argument("--fracs", default="0.01,0.02", help="các biên δ, theo tỉ lệ số nhân TB")
    ap.add_argument("--out", default=os.environ.get("FIG_DIR", "figures") + "/fig_equivalence.pdf")
    a = ap.parse_args()

    # Chỉ vẽ các biên đáng kể (1%, 2%). Thêm 5% chỉ làm trục giãn ra 2,5 lần và
    # bóp khoảng tin cậy thành một vạch không đọc được; 5% để trong phần chữ.
    fracs = [float(x) for x in a.fracs.split(",")]
    deltas = [f * a.gt_mean for f in fracs]
    lim = max(deltas) * 1.45

    fig, ax = plt.subplots(figsize=(6.4, 2.0))

    # Đường dọc chỉ chạy từ y=-0,55 lên: chừa hẳn dải đáy cho nhãn hướng, khỏi đè nhau.
    TOP, BOT = 0.95, -0.55
    ax.fill_betweenx([BOT, TOP], -deltas[-1], deltas[-1], color=ACCENT, alpha=0.06, lw=0)
    # Nhãn δ xếp so le theo chiều dọc; nhãn ngoài cùng quay vào trong để không chạm mép phải.
    for i, (f, d) in enumerate(zip(fracs, deltas)):
        for s in (-1, 1):
            ax.vlines(s * d, BOT, TOP, color=MUTED, ls=(0, (4, 3)), lw=1.0)
        inner = (i == len(fracs) - 1)
        # '%' khong duoc nam trong $...$ (mathtext coi la loi cu phap) -> de ngoai phan toan
        ax.annotate(f"$\\delta$ = {f:.0%} ({d:.2f})",
                    xy=(d + lim * (-0.02 if inner else 0.02), 0.72 - 0.30 * i),
                    fontsize=7.5, color=MUTED, ha="right" if inner else "left", va="center")
    ax.vlines(0, BOT, TOP, color=INK, lw=1.2)

    ax.errorbar(a.dm, 0, xerr=[[a.dm - a.lo], [a.hi - a.dm]], fmt="o", color=ACCENT,
                ecolor=ACCENT, elinewidth=2.0, capsize=4, markersize=7, zorder=5)

    # Số liệu (Δ, CI, δ*) nằm trong chú thích hình, không nhồi vào khung vẽ.
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("$\\Delta$ per-image absolute error,  count-only $-$ mask  (nuclei)", fontsize=9)
    ax.annotate("count-only errs less", xy=(-lim * 0.97, -0.80), fontsize=7.5, color=MUTED,
                ha="left", va="center")
    ax.annotate("mask errs less", xy=(lim * 0.97, -0.80), fontsize=7.5, color=MUTED,
                ha="right", va="center")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", colors=INK, labelsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"[OUT] {a.out}")
    far = max(abs(a.lo), abs(a.hi))          # delta* = dau xa 0 nhat cua CI
    sig = a.lo * a.hi > 0                    # CI khong chua 0
    print(f"  CI {'KHONG chua' if sig else 'chua'} 0 -> khac biet {'CO' if sig else 'KHONG'} y nghia")
    print(f"  delta* = {far:.3f} = {100 * far / a.gt_mean:.2f}% so nhan TB")
    for f, d in zip(fracs, deltas):
        print(f"  delta={f:.0%} ({d:.3f}): {'TUONG DUONG' if far < d else 'chua ket luan duoc'}")


if __name__ == "__main__":
    main()
