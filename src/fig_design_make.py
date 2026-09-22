#!/usr/bin/env python3
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import hypot
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
from matplotlib.image import imread

F = os.environ.get("FIG_DIR", "figures")
INK, MUT, NET = "#222222", "#8A8A8A", "#D8D8D8"
LBL = "#454545"          # ten khoi: dam hon MUT de doc duoc khi in
GRN, BLU, OPS, OPF = "#256B2E", "#1F5FA8", "#7E74A0", "#F4F1FA"
# Chon lam/luc bang do OKLab co mo phong mu mau: lam-luc tach 19,9 va luc-tim(mang) 17,3 duoi deuter.
FS, FSS = 11.0, 9.5          # 9.5 nhan phu / 11 chu giai gradient / 13 tieu de / 17 so chi phi / 19 ky hieu.
# Co chu tinh nguoc tu khô in: hinh rong 9.5in nhung dat o \textwidth (~6.5in) -> thu 0.68 lan.
# 9.5pt o day = 6.5pt tren giay. Ha xuong nua la duoi nguong doc duoc cua Elsevier.

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 600,   # Elsevier: combination art >= 500 dpi
                     "savefig.bbox": "tight", "figure.dpi": 160,
                     # phai tat: mac dinh matplotlib gop moi imshow thanh mot raster khi xuat vector, nen
                     # anh chi con mot lop con vien la patch vector giu zorder rieng -> vien cua to sau
                     # trong chong giay nhay len tren anh cua to truoc. Tat di thi moi anh la mot lop.
                     "image.composite_image": False})
fig, ax = plt.subplots(figsize=(12.0, 6.1))
ax.set_xlim(0, 112); ax.set_ylim(0, 56); ax.set_aspect("equal"); ax.axis("off")


def box(x, y, w, h, fc="#FFFFFF", ec=MUT, lw=1.0, r=0.6, z=2):
    bs = f"round,pad=0,rounding_size={r}" if r > 0 else "square,pad=0"   # r=0 -> goc vuong that
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=bs, fc=fc, ec=ec, lw=lw, zorder=z))


def img(path, x, y, s, ec=None, z=3):
    ax.imshow(imread(f"{F}/{path}"), extent=(x, x + s, y, y + s), zorder=z, aspect="auto")
    if ec:
        ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0,rounding_size=0.01",
                                    fc="none", ec=ec, lw=1.2, zorder=z + 0.5))


def txt(x, y, s, fs=FS, c=INK, w="normal", ha="center", va="center"):
    ax.text(x, y, s, fontsize=fs, color=c, fontweight=w, ha=ha, va=va, zorder=6)


DASH, HL, HW = (0, (2.6, 1.8)), 1.5, 0.5     # dau: dai 1.5 x nua-rong 0.5 = manh, ti le 3:1


def arr(x0, y0, x1, y1, c=MUT, lw=1.1, ls="-", z=1):
    """Than va DAU ve RIENG, DAU la da giac trong TOA DO DU LIEU.

    Hai bay da dinh: (1) FancyArrowPatch ap net dut len ca dau -> dau dut quang va phinh tron;
    (2) marker tam giac dat TAM tai diem cuoi va la tam giac DEU -> nua dau thoc vao trong hop
    va be ra thanh cuc. Ve da giac thi DINH NHON nam dung (x1,y1) va ti le do minh dat."""
    dx, dy = x1 - x0, y1 - y0
    n = hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n                   # huong di; (-uy,ux) la phap tuyen
    bx, by = x1 - ux * HL, y1 - uy * HL       # chan dau mui ten
    ax.plot([x0, bx], [y0, by], color=c, lw=lw, ls=ls, zorder=z,
            solid_capstyle="butt", dash_capstyle="butt")
    ax.add_patch(Polygon([(x1, y1), (bx - uy * HW, by + ux * HW), (bx + uy * HW, by - ux * HW)],
                         closed=True, fc=c, ec="none", zorder=z + 0.1))


def stack(x, y, s, col, path=None, n=2, d=0.8):
    """To giay xep chong SAU tam truoc, lech xuong-phai: nhan khong phai MOT tam ma la ca TAP
    5,179 manh. Lech xuong-phai chu khong len-phai vi phia tren la nhan 'annotation', va cung
    de mui ten roi tu canh trai khong dam vao chong giay.

    To sau phai CUNG LOAI voi to truoc: anh thi ve anh (khong de hop trang rong), so thi de trang.
    zorder < 3 de nam duoi to truoc.

    path la MOT DANH SACH manh KHAC NHAU, khong phai mot manh lap lai: lap lai thi dai lo ra o duoi-phai
    noi tiep dung vien cua to truoc, doc thanh nhan bi vien doi tran ra ngoai khung.

    Buoc z giua cac to phai LON HON 0.5, vi img() ve vien o z+0.5: buoc nho hon thi vien cua to SAU
    dat len anh cua to TRUOC, nhin thanh 'to tren chua vien cua to duoi'."""
    for k in range(n, 0, -1):
        if path:
            img(path[k - 1], x + k * d, y - k * d, s, col, z=3 - 0.8 * k)
        else:
            box(x + k * d, y - k * d, s, s, fc="#FFFFFF", ec=col, lw=1.0, r=0, z=2)


HS = (9.0, 6.8, 4.8, 3.2, 4.8, 6.8, 9.0)     # chieu cao ~ do phan giai, nen lai cho doc duoc
SK = ((0, 6), (1, 5), (2, 4))                # skip
NB, SC = 1.2, 0.82                           # be rong mot thanh; SC ha chieu cao cho can voi feature maps
ARX = 13.6 + NB / 2    # tam thanh dau cua encoder: mui ten gradient dap vao day, nam trai moi skip.
# SUY ra tu NB chu khong viet cung: doi be rong thanh ma quen sua o day thi mui ten dap lech tam.
TOPF, SIDF = "#E6E1F0", "#D6CFE6"        # mat tren / mat ben: dam dan de khoi co chieu sau


def planes(x, y, h, n=3, wf=0.5, sx=0.9, sy=0.9, d=0.8):
    """Feature maps: n KHOI HOP, khong phai tam phang. Moi khoi ve 3 mat - mat truoc (chu nhat),
    mat tren va mat ben (binh hanh, truot (sx,sy)) - to dam dan nen khoi mang CHIEU SAU that.
    Truot (sx,sy) lon hon be rong mat truoc nen khoi nam XEO, nhin ro be rong. wf+sx > d nen cac
    khoi de len nhau; ve tu TRAI sang PHAI de khoi 2 de len khoi 1, khoi 3 de len khoi 2."""
    for k in range(n):
        x0, z = x + k * d, 2 + 0.1 * (k + 1)
        for pts, fc in (([(x0 + wf, y), (x0 + wf + sx, y + sy), (x0 + wf + sx, y + sy + h),
                          (x0 + wf, y + h)], SIDF),
                        ([(x0, y + h), (x0 + wf, y + h), (x0 + wf + sx, y + h + sy),
                          (x0 + sx, y + h + sy)], TOPF)):
            ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=OPS, lw=1.0, zorder=z))
        box(x0, y, wf, h, fc=OPF, ec=OPS, lw=1.0, r=0, z=z + 0.05)


def encdec(x0, y, h, w=17):
    """Khoi encoder-decoder theo quy uoc U-Net: day thanh feature map thap dan roi cao dan, kem skip.

    Khong dung hinh dong ho cat, vi do la quy uoc cua autoencoder, ham y thong tin bi bop qua bottleneck
    va khong co skip. So thanh ve la 7 cho thoang, khong phai 9 muc that (encoder 5 + decoder 4): hinh
    nay la minh hoa kien truc, khong phai dac ta.

    Skip ve DUNG O CANH TREN cua cap thanh doi xung nen tu long vao nhau, va di tu canh PHAI thanh
    encoder sang canh TRAI thanh decoder -> khong cai nao cham cot mui ten gradient (ARX).
    Tra ve (x_cuoi, y_canh_tren_thanh_dau) de mui ten gradient dap dung vao canh."""
    s, yc, f = (w - NB) / (len(HS) - 1), y + h / 2, h * SC / max(HS)   # van can giua o y+h/2
    for k, hk in enumerate(HS):
        box(x0 + k * s, yc - hk * f / 2, NB, hk * f, fc=OPF, ec=OPS, lw=1.1, r=0, z=2)
    for a, b in SK:
        ya = yc + HS[a] * f / 2
        ax.plot([x0 + a * s + NB, x0 + b * s], [ya, ya],
                color=OPS, lw=0.9, ls=(0, (2.2, 1.6)), zorder=2.2)
    return x0 + w, yc + HS[0] * f / 2


def regime(yb, col, tag, title, ann_img, dens, cost, note, grad, tgt=None, sym=None):
    """BAN A: phan DUNG CHUNG (duong xuoi) ep nho, phan KHAC NHAU phong to va THANG COT giua hai hang.

    Toa do giai NGUOC tu rang buoc chu khong dat tay, nen moi mui ten NGANG chi co hai chieu dai: 3,6 cho
    buoc nhay giua hai khoi ke nhau, 10,7 cho quang vuot khoang trong (mui ten 1x1 conv va ca ba mui ten
    giam sat). Bon rang buoc dan toi cap so do:
      o loss thang cot voi o dem  |  o gradient thang cot voi ban do mat do
      nhan '1x1 conv' do duoc rong 8,38 -> mui ten do khong the ngan hon
      dong chu thich duoi chong giay phai nam gon trong be ngang 112
    Doi mot con so thi phai giai lai ca hang, dung sua le."""
    box(0, yb, 112, 26, fc="#FCFCFC", ec=NET, lw=.8, r=1.0, z=0)
    txt(2.5, yb + 23.8, tag, 11, col, "bold", ha="left")
    txt(6.6, yb + 23.8, title, 11, col, "bold", ha="left")   # nhan "(a)" ket thuc o 5,44 -> ho 1,2
    # (truoc la 2,0, doc thanh hai cum roi nhau). Do luon: keo vao con noi rong ho voi "what flows back".

    yf, S = yb + 2.6, 7
    ya = yf + 3.5
    img("fig1_plain.png", 3, yf, S)
    txt(6.5, yf - 1.5, "input patch", FSS, LBL)
    arr(10, ya, 13.6, ya)
    xn, ytop = encdec(13.6, yf, S, w=8)
    txt(17.6, yf - 1.5, "backbone", FSS, LBL)      # mot tu: 'encoder-decoder' phai xuong hai dong moi vua
    arr(xn, ya, 25.2, ya)
    planes(25.2, yf + 1.05, 4.2)
    txt(26.7, yf - 1.5, "features", FSS, LBL)
    arr(28.2, ya, 38.9, ya)
    txt(33.55, ya + 1.7, r"$1\times1$ conv", FSS, LBL)
    img(dens, 38.9, yf, S)
    txt(42.4, yf - 1.5, r"density $\hat d$", FSS, LBL)
    arr(45.9, ya, 49.5, ya)
    ax.add_patch(Circle((51.5, ya), 2.0, fc="#FFFFFF", ec=MUT, lw=1.0, zorder=2))
    txt(51.5, ya, r"$\Sigma$", 10, INK)
    arr(53.5, ya, 57.1, ya)
    box(57.1, yf + 0.5, 10, 6, ec=INK, lw=1.6)
    txt(62.1, ya, r"$\mu=\sum_p \hat d_p$", 12, INK)
    txt(62.1, yf - 1.5, "predicted count", FSS, LBL)

    yl, P = yb + 13.5, 10
    yc = yl + 3.5
    stack(97.5, yl - 1, P, col, path=None if sym else ("fig1_mask1.png", "fig1_mask2.png"))
    if sym:
        box(97.5, yl - 1, P, P, ec=col, lw=1.4, r=0)
        txt(102.5, yl + 4, sym, 19, col, "bold")
    else:
        img(ann_img, 97.5, yl - 1, P, col)
    txt(102.5, yl + 10.1, "annotation", FSS, col, "bold")
    txt(102.5, yl - 4.9, cost, 17, col, "bold")
    txt(102.5, yl - 7.7, note, FSS, LBL)
    # ca hai hang deu co o muc tieu -> ba cap so sanh (chu giai / muc tieu / cai chay ve) thang cot.
    # Cho trong khong doc thanh "khac", no doc thanh "khong co gi". O muc tieu cua hang chi-dem lap lai
    # dung noi dung tam chu giai: do chinh la diem can noi, nhanh mat na phai chuyen doi duong vien thanh
    # ban do day dac, nhanh chi-dem khong co buoc chuyen doi nao.
    if tgt:
        img(tgt, 76.8, yl - 1, P, col)
        txt(81.8, yl + 10.1, "target density", FSS, col, "bold")
    else:
        box(76.8, yl - 1, P, P, fc="#FFFFFF", ec=col, lw=1.4, r=0)
        txt(81.8, yl + 4, sym, 19, col, "bold")
        txt(81.8, yl + 10.1, "target count", FSS, col, "bold")
    arr(97.5, yc, 86.8, yc, col, 1.3, ls=DASH)
    arr(76.8, yc, 66.1, yc, col, 1.3, ls=DASH)
    box(58.1, yl + 1.5, 8, 4, fc="#FFFFFF", ec=col, lw=1.4)
    txt(62.1, yc, "loss", FS, col, "bold")
    xmu = 64.1 if tgt else 62.1
    arr(xmu, yf + 6.5, xmu, yl + 1.5, col, 1.3, ls=DASH)
    if tgt:
        ym = yl - 2.1                 # nut re nhanh dat o duoi mui ten, cach chan dau 1,6: dat giua than
        ax.plot([46.4, 46.4, 60.1], [ya, ym, ym], color=col, lw=1.3, ls=DASH, zorder=1)   # thi cham dau
        arr(60.1, ym, 60.1, yl + 1.5, col, 1.3, ls=DASH)
        ax.add_patch(Circle((46.4, ya), .32, fc=col, ec="none", zorder=4))
    arr(58.1, yc, 47.4, yc, col, 1.3, ls=DASH)
    img(grad, 37.4, yl - 1, P, col)
    # khong kem ky hieu dao ham: chuoi do rong 20,5 trong khi o chi rong 10, nen no tho ra hai ben
    # va cham tieu de hang (b) (chi con ho 1,04). Caption da noi hai o nay trinh bay dai luong nao.
    txt(42.4, yl + 10.1, "what flows back", FSS, col, "bold")
    ax.plot([37.4, ARX, ARX], [yc, yc, yl - 0.93], color=col, lw=1.3, ls=DASH, zorder=1)
    arr(ARX, yl - 0.93, ARX, ytop, col, 1.3, ls=DASH)


regime(29.5, BLU, "(a)", "mask supervision", "fig1_mask.png", "fig1_dens_mask.png",
       "393 h", "expert time\n5,179 patches", "fig1_grad_mask.png", tgt="fig1_density.png")
regime(0.5, GRN, "(b)", "count-only supervision", "fig1_count22.png", "fig1_dens_count.png",
       "48 h", "at most; $\\geq$8$\\times$ less\nthan (a)", "fig1_grad_count.png", sym=r"$N$")


txt(56, 28.0, "or", 13, MUT, "bold")
# Chu giai dau cua gradient, dat mot lan o dai giua hai hang vi no giai nghia cho ca hai o gradient.
# Khong phai thanh mau truu tuong: dau gradient co nghia den, am thi buoc hoc nang mat do tai diem do
# len, duong thi HA xuong. Co no thi o do cua hang chi-dem tu tra loi cau "tai sao do" (lan nay dem ra
# thua nen ha deu khap noi; dem thieu thi ca o se xanh), va o cua hang mat na tu noi "nang len dung cho
# co nhan". Nho vay hinh tu dung duoc, khong bat nguoi doc phai xuong caption moi hieu mau.
for _k, (_c, _lab) in enumerate((("#053061", "raises density here"),
                                 ("#67001F", "lowers density here"))):
    _x = 69 + _k * 21   # dat ben phai: nua trai cua dai giua nam duoi chuoi khoi mang, de bi roi
    box(_x, 27.1, 1.8, 1.8, fc=_c, ec=MUT, lw=0.8, r=0, z=3)
    txt(_x + 2.5, 28.0, _lab, FSS, LBL, ha="left")

fig.savefig(f"{F}/fig_design.pdf")
print("ok")