#!/usr/bin/env python3
import argparse, json, os, sys

import numpy as np
import torch
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "lib"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from dsunet import DensitySigmaUNet  # noqa: E402


def gt_density(polys, size):
    """Ban do mat do dung tu da giac COCO: moi nhan gop 1.0 trai deu tren dien tich cua no."""
    d = np.zeros(size[::-1], np.float32)
    for poly in polys:
        m = Image.new("1", size, 0)
        dr = ImageDraw.Draw(m)
        for p in poly:
            if len(p) >= 6:                # PIL can it nhat 3 dinh; vai da giac COCO bi suy bien
                dr.polygon(list(zip(p[0::2], p[1::2])), fill=1)
        a = np.asarray(m, bool)
        if a.sum():
            d[a] += 1.0 / int(a.sum())
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="thu muc fold kiem tra: images/ + annotations.json")
    ap.add_argument("--ckpt", action="append", default=[], required=True,
                    help='"<hat_giong>=duong/dan.pt", lap lai duoc')
    ap.add_argument("--out", required=True, help="thu muc ghi locuq_s<hat_giong>.npz")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--sigma_mode", default="poisson", choices=["poisson", "nb", "raw"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="0 = het; dat nho de thu nhanh")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = json.load(open(f"{args.coco}/annotations.json"))
    by = {}
    for an in d["annotations"]:
        by.setdefault(an["image_id"], []).append(an["segmentation"])
    imgs = sorted(d["images"], key=lambda i: i["file_name"])
    if args.limit:
        imgs = imgs[:args.limit]
    print(f"[data] {len(imgs)} anh, {sum(len(v) for v in by.values())} nhan da chu thich")

    for spec in args.ckpt:
        seed, path = spec.split("=", 1)
        f = os.path.join(args.out, f"locuq_s{seed}.npz")
        if os.path.exists(f):
            print(f"[s{seed}] da co {f}, bo qua")
            continue
        net = DensitySigmaUNet(ch=args.ch, sigma_mode=args.sigma_mode, backbone=args.backbone)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net = net.to(args.device).eval()

        mu, sig, mass, chance, corr, gt = [], [], [], [], [], []
        with torch.no_grad():
            for k, im in enumerate(imgs):
                polys = by.get(im["id"], [])
                pil = Image.open(f"{args.coco}/images/{im['file_name']}").convert("RGB")
                g = gt_density(polys, pil.size).astype(np.float64)
                x = torch.from_numpy(np.asarray(pil, np.float32) / 255.0)
                x = x.permute(2, 0, 1)[None].to(args.device)
                dens, log_sigma = net(x)
                p = dens[0, 0].cpu().numpy().astype(np.float64)

                mu.append(float(p.sum()))
                sig.append(float(torch.exp(log_sigma)[0]))
                gt.append(len(polys))
                nuc = g > 0
                chance.append(float(nuc.mean()))
                mass.append(float(p[nuc].sum() / p.sum()) if abs(p.sum()) > 1e-8 else np.nan)
                corr.append(float(np.corrcoef(p.ravel(), g.ravel())[0, 1])
                            if p.std() > 1e-8 and g.std() > 1e-8 else np.nan)
                if (k + 1) % 500 == 0:
                    print(f"  [s{seed}] {k+1}/{len(imgs)}", flush=True)
        np.savez(f, mu=mu, sigma=sig, mass=mass, chance=chance, corr=corr, gt=gt)
        print(f"[s{seed}] da luu {f}")


if __name__ == "__main__":
    main()
