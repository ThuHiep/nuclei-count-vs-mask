#!/usr/bin/env python3
import os
import json, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch.nn.functional as F
from PIL import Image, ImageDraw
from dsunet import DensitySigmaUNet

R = os.environ.get("COCO_FOLD3", "data/coco/fold3")
OUT = os.environ.get("FIG_DIR", "figures")
IMG, N = "im0507.png", 22.0
CK = {"mask": os.environ.get("CKPT_MASK", "ckpt/mask_s42.pt"),
      "count": os.environ.get("CKPT_COUNT", "ckpt/count_s42.pt")}

d = json.load(open(f"{R}/annotations.json"))
iid = next(i["id"] for i in d["images"] if i["file_name"] == IMG)
polys = [a["segmentation"] for a in d["annotations"] if a["image_id"] == iid]
im = Image.open(f"{R}/images/{IMG}").convert("RGB")
t = np.zeros((im.height, im.width), np.float32)
for poly in polys:                                   # muc tieu: moi nhan gop 1/dien tich -> tong = so nhan
    m = Image.new("1", im.size, 0)
    for p in poly:
        ImageDraw.Draw(m).polygon(list(zip(p[0::2], p[1::2])), fill=1)
    a = np.asarray(m, bool)
    if a.sum():
        t[a] += 1.0 / int(a.sum())
assert abs(t.sum() - len(polys)) < 0.5
T = torch.from_numpy(t)[None, None]
x = torch.from_numpy(np.asarray(im, np.float32) / 255.).permute(2, 0, 1)[None]


def net_of(tag):
    n = DensitySigmaUNet(ch=32, sigma_mode="poisson", backbone="efficientnet_lite0").eval()
    n.load_state_dict(torch.load(CK[tag], map_location="cpu"))
    return n


# vien nhan that, ve chong len ca hai ban do -> doc duoc "khoi co roi dung cho khong"
nuc = t > 0
edge = np.zeros_like(nuc)
for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    edge |= nuc & ~np.roll(nuc, (dy, dx), (0, 1))

for tag in CK:                                       # ban do mat do du doan cua tung nhanh
    with torch.no_grad():
        p = net_of(tag)(x)[0][0, 0].numpy()
    u = np.clip(p / np.percentile(p[p > 0], 96), 0, 1) ** 0.35    # >99% diem = 0 -> gamma de thay chi tiet
    rgb = (plt.get_cmap("magma")(u)[..., :3] * 255).astype(np.uint8)
    rgb[edge] = (90, 220, 255)
    Image.fromarray(rgb).save(f"{OUT}/fig1_dens_{tag}.png")
    frac = p[nuc].sum() / p.sum() / nuc.mean()
    print(f"{tag}: mu = {p.sum():.2f} | ti so khoi trong nhan = {frac:.2f}")

net = net_of("count")                                # cung mot bo trong so cho ca hai gradient
for tag, fn in (("mask", lambda D: F.mse_loss(D, T)), ("count", lambda D: (D.sum() - N).abs())):
    D = net(x)[0]; D.retain_grad(); fn(D).backward()
    g = D.grad[0, 0].numpy()
    v = np.percentile(np.abs(g), 99.0) or np.abs(g).max()
    u = np.sign(g) * np.clip(np.abs(g) / v, 0, 1) ** 0.35   # gamma giu dau, nhu o mat do: khoi nhan bao hoa
    plt.imsave(f"{OUT}/fig1_grad_{tag}.png", u, cmap="RdBu_r", vmin=-1, vmax=1)
    print(f"grad {tag}: {len(np.unique(g))} gia tri khac nhau | thang +-{v:.2e}")
