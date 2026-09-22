#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
from PIL import Image, ImageDraw
from dsunet import DensitySigmaUNet

R = os.environ.get("COCO_FOLD3", "data/coco/fold3")
OUT = os.environ.get("FIG_DIR", "figures")
IMG, W = "im0507.png", 512
CKPT = os.environ.get("CKPT_MASK", "ckpt/mask_s42.pt")
ANCH = np.array([[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]], float)

im = Image.open(f"{R}/images/{IMG}").convert("RGB")
d = json.load(open(f"{R}/annotations.json"))
iid = next(i["id"] for i in d["images"] if i["file_name"] == IMG)
polys = [a["segmentation"] for a in d["annotations"] if a["image_id"] == iid]

# thang mau lay tu GT (giong fig_design_patches.py) de hai anh cung don vi
gt = np.zeros((im.height, im.width), np.float32)
for poly in polys:
    m = Image.new("1", im.size, 0)
    for p in poly:
        ImageDraw.Draw(m).polygon(list(zip(p[0::2], p[1::2])), fill=1)
    mr = np.asarray(m, bool)
    if mr.sum():
        gt[mr] += 1.0 / int(mr.sum())
scale = np.percentile(gt[gt > 0], 99)

net = DensitySigmaUNet(ch=32, sigma_mode="poisson", backbone="efficientnet_lite0").eval()
net.load_state_dict(torch.load(CKPT, map_location="cpu"))
x = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)[None]
with torch.no_grad():
    dens = net(x)[0][0, 0].numpy()

u = np.clip(dens / scale, 0, 1) ** 0.7
rgb = np.stack([np.interp(u, np.linspace(0, 1, len(ANCH)), ANCH[:, c]) for c in range(3)], -1)
Image.fromarray(rgb.astype(np.uint8)).resize((W, W), Image.LANCZOS).save(f"{OUT}/fig1_pred.png")
print(f"{IMG}: GT {len(polys)} nhan (tong {gt.sum():.2f}) | du doan {dens.sum():.2f}")
