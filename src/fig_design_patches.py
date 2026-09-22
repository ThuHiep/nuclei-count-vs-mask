#!/usr/bin/env python3
import os
import json, collections
from PIL import Image, ImageDraw

R = os.environ.get("COCO_FOLD3", "data/coco/fold3")
OUT = os.environ.get("FIG_DIR", "figures")
IMG, K, W = "im0507.png", 4, 512          # K=he so sieu lay mau, W=be rong xuat
# Chong giay trong Hinh 1 nghia la 5.179 manh khac nhau. Ve lai cung mot manh thi dai lo ra o duoi-phai
# noi tiep dung viend cua manh truoc -> doc thanh nhan bi vien doi tran ra ngoai khung. Hai manh nay cung
# fold 3, cung 22 nhan nhu im0507 nen chong giay trong dong deu.
IMG2 = ("im0038.png", "im0179.png")

d = json.load(open(f"{R}/annotations.json"))
iid = next(i["id"] for i in d["images"] if i["file_name"] == IMG)
polys = [a["segmentation"] for a in d["annotations"] if a["image_id"] == iid]

im = Image.open(f"{R}/images/{IMG}").convert("RGB")
im.resize((W, W), Image.LANCZOS).save(f"{OUT}/fig1_plain.png")

def outlined(fn, out):
    p2 = [a["segmentation"] for a in d["annotations"]
          if a["image_id"] == next(i["id"] for i in d["images"] if i["file_name"] == fn)]
    b = Image.open(f"{R}/images/{fn}").convert("RGB")
    b = b.resize((b.width * K, b.height * K), Image.LANCZOS)
    dr = ImageDraw.Draw(b)
    for poly in p2:
        for p in poly:
            pts = [(x * K, y * K) for x, y in zip(p[0::2], p[1::2])]
            # vang, khong phai luc: luc la mau hang (b) -> mot mau se mang hai nghia nguoc nhau.
            dr.line(pts + [pts[0]], fill=(255, 224, 0), width=2 * K, joint="curve")
    b.resize((W, W), Image.LANCZOS).save(f"{OUT}/{out}.png")


outlined(IMG, "fig1_mask")
for k, fn in enumerate(IMG2, 1):
    outlined(fn, f"fig1_mask{k}")

# --- ban do mat do muc tieu: moi nhan gop 1/dien tich -> tong = dung so nhan ---
# cung cong thuc dsunet.py:189 (dens[mr] += 1.0/a), khong phai Gaussian o tam
import numpy as np

dens = np.zeros((im.height, im.width), np.float32)
for poly in polys:
    m = Image.new("1", im.size, 0)
    for p in poly:
        ImageDraw.Draw(m).polygon(list(zip(p[0::2], p[1::2])), fill=1)
    mr = np.asarray(m, bool)
    if mr.sum():
        dens[mr] += 1.0 / int(mr.sum())
assert abs(dens.sum() - len(polys)) < 0.5, f"tong ban do {dens.sum():.2f} != {len(polys)} nhan"

ANCH = np.array([[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]], float)
# chi de hien thi: chuan hoa theo phan vi 99, khong theo max. Anh nay co 1 nhan 9 px cho 1/a
# lon gap 159 lan cac nhan khac -> lay max thi ca ban do den thui. Gia tri that van tong = 22.
u = np.clip(dens / np.percentile(dens[dens > 0], 99), 0, 1) ** 0.7
rgb = np.stack([np.interp(u, np.linspace(0, 1, len(ANCH)), ANCH[:, c]) for c in range(3)], -1)
Image.fromarray(rgb.astype(np.uint8)).resize((W, W), Image.LANCZOS).save(f"{OUT}/fig1_density.png")

print(f"{IMG}: {len(polys)} nhan | tong ban do mat do = {dens.sum():.3f} -> 3 anh trong {OUT}/")
