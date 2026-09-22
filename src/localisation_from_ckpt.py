#!/usr/bin/env python3
import argparse, json, os
import numpy as np
import torch
from PIL import Image, ImageDraw
from dsunet import DensitySigmaUNet


def gt_density(polys, size):
    d = np.zeros(size[::-1], np.float32)
    for poly in polys:
        m = Image.new("1", size, 0)
        dr = ImageDraw.Draw(m)
        for p in poly:
            if len(p) >= 6:                       # PIL can >=3 dinh; vai da giac COCO bi suy bien
                dr.polygon(list(zip(p[0::2], p[1::2])), fill=1)
        a = np.asarray(m, bool)
        if a.sum():
            d[a] += 1.0 / int(a.sum())
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="thu muc fold3: images/ + annotations.json")
    ap.add_argument("--ckpt", action="append", default=[], help='"ten=duong/dan.pt", lap lai duoc')
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--limit", type=int, default=0, help="0 = het; dat nho de thu nhanh")
    ap.add_argument("--out_csv", default="")
    a = ap.parse_args()

    d = json.load(open(f"{a.coco}/annotations.json"))
    by = {}
    for an in d["annotations"]:
        by.setdefault(an["image_id"], []).append(an["segmentation"])
    imgs = sorted(d["images"], key=lambda i: i["file_name"])
    if a.limit:
        imgs = imgs[:a.limit]
    print(f"[data] {len(imgs)} anh | {sum(len(v) for v in by.values())} nhan")

    dev = "cpu"
    rows = []
    for name_path in a.ckpt:
        name, path = name_path.split("=", 1)
        net = DensitySigmaUNet(ch=a.ch, sigma_mode="poisson", backbone=a.backbone).eval().to(dev)
        net.load_state_dict(torch.load(path, map_location=dev))
        corrs, mass, chance, pr, yt = [], [], [], [], []
        with torch.no_grad():
            for k, im in enumerate(imgs):
                polys = by.get(im["id"], [])
                pil = Image.open(f"{a.coco}/images/{im['file_name']}").convert("RGB")
                g = gt_density(polys, pil.size).astype(np.float64)
                x = torch.from_numpy(np.asarray(pil, np.float32) / 255.0).permute(2, 0, 1)[None]
                p = net(x)[0][0, 0].numpy().astype(np.float64)
                pr.append(float(p.sum())); yt.append(len(polys))
                nuc = g > 0
                chance.append(float(nuc.mean()))
                if p.std() > 1e-8 and g.std() > 1e-8:
                    corrs.append(float(np.corrcoef(p.ravel(), g.ravel())[0, 1]))
                if abs(p.sum()) > 1e-8:
                    mass.append(float(p[nuc].sum() / p.sum()))
                if (k + 1) % 400 == 0:
                    print(f"  {name}: {k+1}/{len(imgs)}", flush=True)
        pr, yt = np.array(pr), np.array(yt)
        r2 = 1 - ((yt - pr) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        ch, ms = float(np.mean(chance)), float(np.mean(mass))
        rows.append((name, ms / ch, float(np.std(mass) / ch), float(np.mean(corrs)), r2, ch))
        print(f"[{name}] ratio {ms/ch:.3f} | corr {np.mean(corrs):+.4f} | count-R2 {r2:+.4f}")

    print(f"\n  chance (nhan chiem) = {rows[0][5]:.3f} | ban do GT hoan hao = {1/rows[0][5]:.2f}x | deu = 1.00x")
    print(f"  {'giam sat':16s} {'ratio (dinh vi)':>18s} {'corr':>10s} {'count-R2':>10s}")
    for n, r, sd, c, r2, _ in rows:
        print(f"  {n:16s} {r:>13.3f}+-{sd:.3f} {c:>+10.4f} {r2:>+10.4f}")
    if a.out_csv:
        with open(a.out_csv, "w") as f:
            f.write("arm,ratio,ratio_sd,corr,count_r2,chance\n")
            for r in rows:
                f.write(",".join(str(x) for x in (r[0], r[1], r[2], r[3], r[4], r[5])) + "\n")


if __name__ == "__main__":
    main()
