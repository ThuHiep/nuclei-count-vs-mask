#!/usr/bin/env python3
import argparse, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "lib"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dsunet import build_pannuke_density, train
from losses import count_from_density


@torch.no_grad()
def predict_count(model, img_uint8, dev):
    x = torch.from_numpy(img_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
    return float(count_from_density(model(x)[0])[0])


def enumerate_instances(masks):
    """masks: 5 kênh label (mỗi id != 0 = 1 nhân). -> list boolean mask từng nhân."""
    inst = []
    for k in range(5):
        lab = np.asarray(masks[k])
        for iid in np.unique(lab):
            if iid == 0:
                continue
            inst.append(lab == iid)
    return inst


def dilate_masks(inst, k):
    """NỞ mask nhân k pixel. Mask PanNuke thường KHÔNG phủ hết viền tối của nhân -> gỡ xong còn
    VÀNH nhân, model vẫn thấy bằng chứng nhân => dốc đáp ứng bị kéo xuống oan. k=2 phủ viền mà
    chưa ăn nhiều vào mô. k=0 = giữ hành vi cũ (để so sánh)."""
    if k <= 0:
        return inst
    from scipy.ndimage import binary_dilation
    st = np.ones((3, 3), bool)
    return [binary_dilation(b, st, iterations=k) for b in inst]


def shift_to_background(b, uni, rng, tries=40):
    """SHAM-CONTROL: dịch mask nhân b sang vị trí NỀN (không đè nhân nào), GIỮ NGUYÊN hình dạng+diện tích.
    Dùng để tách "model đáp ứng NỘI DUNG NHÂN" khỏi "model đáp ứng ARTIFACT của inpaint":
    gỡ cùng số vùng, cùng diện tích, cùng kiểu inpaint — chỉ khác vị trí có/không có nhân.
    None nếu ảnh quá dày nhân, không còn chỗ nền."""
    ys, xs = np.where(b)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    patch = b[y0:y1 + 1, x0:x1 + 1]
    h, w = patch.shape; H, W = b.shape
    if h >= H or w >= W:
        return None
    for _ in range(tries):
        ny = int(rng.integers(0, H - h + 1)); nx = int(rng.integers(0, W - w + 1))
        if not (uni[ny:ny + h, nx:nx + w] & patch).any():
            out = np.zeros_like(b); out[ny:ny + h, nx:nx + w] = patch
            return out
    return None


_INPAINT = None   # đường thực sự dùng: "telea" hoặc "median-fill" — phải biết, vì artifact khác nhau xa


def occlude(img, remove_masks, mode="telea", uni=None, rng=None):
    """Gỡ vùng trong remove_masks. 3 cách điền, KHÁC NHAU ở chỗ có thể mang texture NHÂN vào không:
      telea    : cv2.inpaint lan mô xung quanh. Tự nhiên NHƯNG quanh 1 nhân là các NHÂN LÂN CẬN
                 -> có thể lôi texture nhân vào lỗ => kéo dốc đáp ứng xuống một cách nhân tạo.
                 (sham KHÔNG bắt được lỗi này: lỗ của sham nằm giữa nền nên chỉ được điền bằng nền.)
      bgpatch  : copy một mảng mô CÙNG HÌNH DẠNG lấy từ vùng KHÔNG có nhân của chính ảnh đó.
                 Giữ thống kê texture mô, KHÔNG mang nội dung nhân => control TỐT NHẤT cho dốc.
      bgmedian : điền màu nền trung vị (đĩa phẳng, rất lộ). Chặn trên về "lỗ hoàn toàn không có nhân".
    """
    global _INPAINT
    if not remove_masks:
        return img.copy()
    m = np.zeros(img.shape[:2], np.uint8)
    for b in remove_masks:
        m[b] = 1
    if mode == "bgpatch":
        out = img.copy()
        bg = (np.median(img[m == 0].reshape(-1, 3), axis=0).astype(np.uint8)
              if (m == 0).any() else img.reshape(-1, 3).mean(0).astype(np.uint8))
        for b in remove_masks:
            src = shift_to_background(b, uni if uni is not None else m.astype(bool), rng)
            # src là b dịch tịnh tiến -> cùng số pixel, cùng thứ tự row-major => gán 1-1 đúng
            out[b] = img[src] if src is not None else bg
        if _INPAINT is None:
            _INPAINT = "bgpatch"; print("[occlude] điền BGPATCH (copy mô từ vùng không có nhân, cùng hình dạng)")
        return out
    if mode == "bgmedian":
        out = img.copy()
        bg = (np.median(img[m == 0].reshape(-1, 3), axis=0) if (m == 0).any() else img.reshape(-1, 3).mean(0))
        out[m == 1] = bg.astype(np.uint8)
        if _INPAINT is None:
            _INPAINT = "bgmedian"; print("[occlude] điền BGMEDIAN (đĩa phẳng màu nền trung vị)")
        return out
    try:
        import cv2
        out = cv2.inpaint(np.ascontiguousarray(img), m, 3, cv2.INPAINT_TELEA)
        if _INPAINT is None:
            _INPAINT = "telea"; print("[occlude] dùng cv2.inpaint TELEA (mô nền lan vào) — đúng thiết kế")
        return out
    except Exception as e:
        if _INPAINT is None:
            _INPAINT = "median-fill"
            print(f"[occlude] cv2 KHÔNG dùng được ({type(e).__name__}: {e}) -> fallback ĐIỀN MÀU TRUNG VỊ. "
                  f"Vùng gỡ thành đĩa phẳng rất lộ => artifact MẠNH hơn Telea, đọc phải thận trọng hơn.")
        out = img.copy()
        bg = np.median(img[m == 0].reshape(-1, 3), axis=0) if (m == 0).any() else img.reshape(-1, 3).mean(0)
        out[m == 1] = bg.astype(np.uint8)
        return out


def save_panel(path, panes):
    """panes = [(ảnh uint8, tiêu đề), ...] -> 1 hàng ảnh cạnh nhau kèm count dự đoán.
    Để MẮT kiểm: gỡ nhân thì count có tụt? sham (gỡ nền) có làm count tụt oan? inpaint có để lại
    vệt giống nhân? — trả lời được bằng cách NHÌN, không chỉ bằng số."""
    n = len(panes)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.5))
    for a, (im, t) in zip(np.atleast_1d(axes).ravel(), panes):
        a.imshow(im); a.set_title(t, fontsize=8); a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def load_ckpt_any(path, map_location):
    """Nạp ckpt kể cả khi nó đã bị GIẢI NÉN thành thư mục. File .pt là một zip, nên Safari (macOS,
    'Open safe files after downloading') hoặc trình upload dataset có thể bung nó ra thành
    <name>/{data.pkl, data/, version, ...}. Gặp thư mục thì nén lại trong bộ nhớ rồi nạp.
    Lưu ý: entry trong .pt có mtime TRƯỚC 1980 -> zipfile.write() báo lỗi, phải dùng writestr
    với date_time cố định."""
    if os.path.isdir(path):
        import io, zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for r, _, fs in os.walk(path):
                for f in sorted(fs):
                    p = os.path.join(r, f)
                    zi = zipfile.ZipInfo(os.path.join("ckpt", os.path.relpath(p, path)),
                                         date_time=(1980, 1, 1, 0, 0, 0))
                    with open(p, "rb") as fh:
                        z.writestr(zi, fh.read())
        buf.seek(0)
        print(f"[ckpt] '{path}' là THƯ MỤC (đã bị giải nén) -> nén lại trong RAM rồi nạp")
        return torch.load(buf, map_location=map_location)
    return torch.load(path, map_location=map_location)


def r2_mae(pred, gt):
    ss = ((gt - pred) ** 2).sum(); st = ((gt - gt.mean()) ** 2).sum()
    return (1 - ss / st if st > 0 else float("nan")), float(np.abs(pred - gt).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", required=True)
    ap.add_argument("--train_folds", default="1,2")
    ap.add_argument("--test_fold", default="3")
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preload", action="store_true",
                    help="đẩy tập train lên GPU 1 lần -> train nhanh ~5-15x (PanNuke fold12 ~2.4GB, vừa T4).")
    ap.add_argument("--n_occ", type=int, default=300, help="số ảnh test dùng cho phân tích gỡ-nhân (trải đều)")
    ap.add_argument("--fracs", default="0,0.1,0.2,0.3,0.5,0.75,1.0",
                    help="tỉ lệ nhân bị gỡ. CÓ liều NHỎ (0.1-0.3): ở đó ảnh còn gần phân phối train nên "
                         "độ nhạy/nhân đáng tin; liều 1.0 gỡ 30-40%% diện tích -> lệch phân phối mạnh.")
    ap.add_argument("--reps", type=int, default=3, help="số tập con ngẫu nhiên mỗi tỉ lệ (giảm nhiễu)")
    ap.add_argument("--no_sham", action="store_true",
                    help="tắt sham-control (gỡ vùng NỀN cùng diện tích/hình dạng). MẶC ĐỊNH BẬT — không có nó "
                         "thì không tách được đáp-ứng-nhân khỏi artifact inpaint (count gỡ-hết 0.705 vô nghĩa).")
    ap.add_argument("--ckpt", default="",
                    help="đường dẫn .pt: có sẵn thì nạp, chưa có thì huấn luyện rồi lưu. "
                         "Cho phép chạy lại phân tích nhiều lần trên ĐÚNG một model.")
    ap.add_argument("--out_dir", default="evidence_out",
                    help="nơi lưu PNG/npz. dat ngoai kho ma nguon (dat ngoai kho ma nguon) — mặc định tương đối "
                         "nên rơi VÀO repo và bị 'rm -rf <repo>' xóa mất hình.")
    ap.add_argument("--n_vis", type=int, default=12,
                    help="số ảnh lưu PANEL minh họa vào <out_dir>/samples/ (gốc | gỡ 30%% | gỡ hết | sham) "
                         "kèm count dự đoán -> xem MẮT THẤY model phản ứng thế nào. 0 = tắt.")
    ap.add_argument("--dilate", type=int, default=2,
                    help="nở mask nhân bao nhiêu pixel trước khi gỡ. Mask PanNuke hụt viền tối của nhân -> "
                         "không nở thì gỡ xong còn VÀNH nhân. 0 = hành vi cũ.")
    ap.add_argument("--fill", default="telea", choices=["telea", "bgpatch", "bgmedian"],
                    help="cách điền chỗ vừa gỡ. telea=inpaint (có thể LÔI texture nhân lân cận vào -> dốc bị "
                         "kéo xuống); bgpatch=copy mô từ vùng không nhân, cùng hình dạng (KHÔNG mang nội dung "
                         "nhân -> đo dốc SẠCH nhất); bgmedian=đĩa phẳng màu nền. Chạy telea VÀ bgpatch rồi so.")
    ap.add_argument("--min_effect", type=float, default=0.02,
                    help="ngưỡng EFFECT SIZE để dám gọi 'đặc hiệu với nhân' (đơn vị count chuẩn hóa). "
                         "p<0.05 mà effect≈0 thì KHÔNG phải bằng chứng — n lớn làm p nhỏ vô nghĩa.")
    ap.add_argument("--zip", action="store_true",
                    help="nén <out_dir> thành .zip cạnh nó (tien tai mot lan).")
    ap.add_argument("--work_dir", default="work",
                    help="cache GT-density. dat ngoai kho ma nguon (dat ngoai kho ma nguon) để tái dùng "
                         "cache parity_pannuke đã build + không bị 'rm -rf <repo>' xóa.")
    ap.add_argument("--deterministic", action="store_true",
                    help="bật cuDNN tất định — CHẬM hơn nhiều với efflite0 trên PanNuke lớn. Mặc định TẮT.")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir = os.path.abspath(args.out_dir)         # đổi sang đường dẫn tuyệt đối cho dễ tìm
    vis_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(vis_dir, exist_ok=True)
    try:                                       # work_dir co the chi doc
        os.makedirs(args.work_dir, exist_ok=True)   # (cache pkl có sẵn -> chỉ đọc, không cần ghi)
    except OSError as e:
        print(f"[work_dir] không tạo được ({e}) — chỉ đọc cache")
    if args.out_dir.startswith(_REPO + os.sep):
        print(f"[CẢNH BÁO] out_dir nằm TRONG repo ({args.out_dir}) -> 'rm -rf <repo>' lần clone sau sẽ XÓA "
              f"het hinh. Dat --out_dir ra ngoai kho ma nguon.")
    print(f"[out] hình + npz sẽ lưu vào: {args.out_dir}  (panel ảnh: {vis_dir})")

    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True   # nhanh; PanNuke lớn không cần tất định

    # ---------- train count-only (headline) ----------
    tr_folds = [int(x) for x in args.train_folds.split(",")]
    fstr = "".join(str(x) for x in sorted(tr_folds))
    train_data = build_pannuke_density(args.pannuke_root, tr_folds, dev,
                                       f"{args.work_dir}/gt_density_pannuke_f{fstr}.pkl")
    test_data = build_pannuke_density(args.pannuke_root, [int(args.test_fold)], dev,
                                      f"{args.work_dir}/gt_density_pannuke_f{args.test_fold}.pkl")
    data = list(train_data) + list(test_data)
    tr = list(range(len(train_data))); te = list(range(len(train_data), len(data)))
    print(f"[data] train fold{tr_folds}={len(tr)} / test fold{args.test_fold}={len(te)}")
    if args.ckpt and os.path.exists(args.ckpt):
        from dsunet import DensitySigmaUNet
        model = DensitySigmaUNet(args.ch, "poisson", args.backbone).to(dev)
        model.load_state_dict(load_ckpt_any(args.ckpt, dev))
        print(f"[ckpt] NẠP {args.ckpt} (bỏ qua train)")
    else:
        model = train(data, dev, args.epochs, args.ch, args.lr, tr,
                      0.0, 1.0, 0.01, 0.5, 16, True, "poisson", args.backbone, preload=args.preload)  # count-only
        if args.ckpt:
            torch.save(model.state_dict(), args.ckpt); print(f"[ckpt] LƯU {args.ckpt}")
    model.eval()

    # ---------- scatter Pred-vs-GT + per-organ ----------
    pred = np.array([predict_count(model, data[i]["img"], dev) for i in te])
    gt = np.array([data[i]["gt"] for i in te])
    organ = np.array([data[i]["organ"] for i in te])
    R2, MAE = r2_mae(pred, gt)
    print(f"\nPred-vs-GT (count-only, fold{args.test_fold}, n={len(te)}): R²={R2:+.4f}  MAE={MAE:.3f}")
    print(f"  {'organ':22s} {'n':>4s} {'R²':>8s} {'MAE':>7s}")
    for org in sorted(set(organ)):
        mk = organ == org
        if mk.sum() >= 5:
            r2o, maeo = r2_mae(pred[mk], gt[mk])
            print(f"  {org:22s} {mk.sum():4d} {r2o:+8.3f} {maeo:7.2f}")
    np.savez(f"{args.out_dir}/preds.npz", pred=pred, gt=gt, organ=organ, R2=R2, MAE=MAE)

    organs = sorted(set(organ)); cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(5, 5))
    for j, org in enumerate(organs):
        mk = organ == org
        plt.scatter(gt[mk], pred[mk], s=8, alpha=0.5, color=cmap(j % 20), label=org)
    lim = max(gt.max(), pred.max()) * 1.05
    plt.plot([0, lim], [0, lim], "k--", lw=1)
    plt.xlim(0, lim); plt.ylim(0, lim)
    plt.xlabel("Số đếm thực (GT)"); plt.ylabel("Số đếm dự đoán (count-only)")
    plt.title(f"Pred-vs-GT  R²={R2:.3f}  MAE={MAE:.2f}")
    plt.legend(fontsize=5, ncol=2, markerscale=1.5, loc="upper left")
    plt.tight_layout(); plt.savefig(f"{args.out_dir}/scatter.png", dpi=160); plt.close()

    # Bland-Altman
    mean_c = (pred + gt) / 2; diff = pred - gt
    bias = diff.mean(); sd = diff.std()
    plt.figure(figsize=(5, 4))
    plt.scatter(mean_c, diff, s=8, alpha=0.4, color="steelblue")
    for y, ls in [(bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")]:
        plt.axhline(y, color="crimson", ls=ls, lw=1)
    plt.xlabel("Trung bình (pred, GT)"); plt.ylabel("Hiệu (pred − GT)")
    plt.title(f"Bland–Altman  bias={bias:+.2f}  ±1.96σ={1.96*sd:.1f}")
    plt.tight_layout(); plt.savefig(f"{args.out_dir}/bland_altman.png", dpi=160); plt.close()
    print(f"  Bland-Altman bias={bias:+.3f} (≈0 = không thiên vị hệ thống), 1.96σ={1.96*sd:.2f}")

    # ---------- dose-response gỡ nhân ----------
    from pannuke_loader import PanNukeFold
    pf = PanNukeFold(args.pannuke_root, int(args.test_fold))
    idx_occ = np.linspace(0, len(pf) - 1, min(args.n_occ, len(pf))).astype(int)
    fracs = [float(x) for x in args.fracs.split(",")]
    rng = np.random.default_rng(args.seed)
    vrng = np.random.default_rng(args.seed + 1000)   # rng riêng cho panel minh họa
    rows = []   # (img_id, frac, N, remaining_true, pred, full_pred, pred_sham, n_sham_placed)
    for c, i in enumerate(idx_occ):
        s = pf[int(i)]
        if s["masks"] is None:            # masks.npy đã xoá -> không gỡ-nhân được
            continue
        img = np.ascontiguousarray(np.asarray(s["image"])[..., :3]).astype(np.uint8)
        inst = dilate_masks(enumerate_instances(s["masks"]), args.dilate)
        N = len(inst)
        if N == 0:
            continue
        uni = np.zeros(img.shape[:2], bool)
        for b in inst:
            uni |= b
        full_pred = predict_count(model, img, dev)

        # ---- panel ảnh minh họa (rng riêng -> không làm lệch chuỗi ngẫu nhiên của phép đo chính) ----
        if c < args.n_vis:
            panes = [(img, f"gốc: GT={N}  pred={full_pred:.1f}")]
            for fv in (0.3, 1.0):
                kk = int(round(fv * N))
                s2 = vrng.choice(N, size=kk, replace=False) if kk else np.array([], int)
                oc = occlude(img, [inst[j] for j in s2], args.fill, uni, vrng)
                panes.append((oc, f"gỡ {fv*100:.0f}% NHÂN (còn {N-kk})\npred={predict_count(model, oc, dev):.1f}"))
            if not args.no_sham:
                shm = [m for m in (shift_to_background(b, uni, vrng) for b in inst) if m is not None]
                oc = occlude(img, shm, args.fill, uni, vrng)
                panes.append((oc, f"SHAM: gỡ {len(shm)} vùng NỀN (nhân còn {N})\npred={predict_count(model, oc, dev):.1f}"))
            save_panel(os.path.join(vis_dir, f"panel_{int(i):04d}.png"), panes)

        for frac in fracs:
            krm = int(round(frac * N))
            for _ in range(args.reps if 0 < frac < 1 else 1):   # frac 0/1 xác định -> khỏi lặp
                sel = rng.choice(N, size=krm, replace=False) if krm > 0 else np.array([], int)
                sel_m = [inst[j] for j in sel]
                a_nuc = int(np.logical_or.reduce(sel_m).sum()) if sel_m else 0
                p = predict_count(model, occlude(img, sel_m, args.fill, uni, rng), dev)
                # sham: gỡ cùng số vùng, cùng hình dạng, cùng diện tích, cùng inpaint — chỉ khác: ở nền.
                # busy tích lũy để sham không đè nhau (nếu đè -> diện tích gỡ thực nhỏ hơn nhân -> so sánh
                # thành ra bất công và sẽ kết luận sai). a_sh cho phép kiểm tra diện tích có khớp thật không.
                p_sh, n_sh, a_sh = full_pred, 0, 0
                if not args.no_sham and krm > 0:
                    busy = uni.copy(); shm = []
                    for j in sel:
                        m2 = shift_to_background(inst[j], busy, rng)
                        if m2 is not None:
                            shm.append(m2); busy |= m2
                    n_sh = len(shm)
                    if n_sh:
                        a_sh = int(np.logical_or.reduce(shm).sum())
                        p_sh = predict_count(model, occlude(img, shm, args.fill, uni, rng), dev)
                    else:
                        p_sh = float("nan")
                rows.append((c, frac, N, N - krm, p, full_pred, p_sh, n_sh, a_nuc, a_sh))
        if (c + 1) % 50 == 0:
            print(f"  [occ] {c+1}/{len(idx_occ)} ảnh")
    rows = np.array(rows, float)
    img_a, frac_a, N_a, rem_a, pred_a, full_a, sham_a, nsh_a, anuc_a, ash_a = rows.T
    ar_a = np.divide(ash_a, anuc_a, out=np.ones_like(ash_a), where=anuc_a > 0)   # diện tích sham / nhân
    ok = full_a > 1e-6
    norm = np.where(ok, pred_a / np.where(ok, full_a, 1), np.nan)      # count sau gỡ nhân / count gốc
    norm_s = np.where(ok, sham_a / np.where(ok, full_a, 1), np.nan)    # ... sau gỡ nền (sham)
    keep = 1 - frac_a
    cov = float(np.nanmean(anuc_a[frac_a == 1.0])) / (256 * 256) if (frac_a == 1.0).any() else float("nan")
    print(f"\nDose-response gỡ nhân (n_ảnh={len(set(img_a.astype(int)))}, {len(rows)} phép đo, "
          f"fill={args.fill}, dilate={args.dilate}px, gỡ-hết phủ {cov:.0%} diện tích ảnh):")

    # (a) độ dốc trong-ảnh — thống kê quyết định. (Hồi quy gộp mọi ảnh bị chi phối bởi biến thiên N
    # giữa các ảnh -> chỉ lặp lại finding , không đo nhân quả. Xem (d), chỉ để tham chiếu.)
    sl_in = []
    for g in sorted(set(img_a)):
        m = (img_a == g) & np.isfinite(norm)
        if len(set(keep[m])) >= 3:
            A = np.vstack([keep[m], np.ones(m.sum())]).T
            sl_in.append(np.linalg.lstsq(A, norm[m], rcond=None)[0][0])
    sl_in = np.array(sl_in)
    print(f"  (a) độ dốc TRONG-ẢNH (norm vs tỉ lệ nhân còn lại) = {sl_in.mean():.3f}±{sl_in.std():.3f} "
          f"(n={len(sl_in)} ảnh)  |  ĐẾM TRUNG THỰC => 1.0")

    # (b) độ nhạy mỗi nhân ở liều nhỏ (frac<=0.3: ảnh còn gần phân phối train -> đáng tin nhất)
    sm = (frac_a > 0) & (frac_a <= 0.3) & np.isfinite(norm)
    krm_a = N_a - rem_a
    sens = ((full_a - pred_a) / np.maximum(krm_a, 1))[sm]
    print(f"  (b) Δcount MỖI nhân bị gỡ (liều nhỏ ≤30%) = {sens.mean():+.3f}±{sens.std():.3f}  "
          f"(=1.0 nghĩa gỡ 1 nhân thì count tụt đúng 1)")

    # (c) sham-control: tách đáp-ứng-nhân khỏi artifact inpaint
    xs = sorted(set(keep)); mean_norm, sd_norm, mean_sh = [], [], []
    if args.no_sham:
        for k in xs:
            m = (keep == k); mean_norm.append(np.nanmean(norm[m])); sd_norm.append(np.nanstd(norm[m])); mean_sh.append(np.nan)
        print("  (c) không có đối chứng giả (--no_sham) nên không tách được ảnh hưởng của phép lấp lỗ.")
    else:
        print(f"  (c) sham-control (gỡ NỀN cùng diện tích/hình dạng) theo liều:")
        print(f"      {'gỡ':>5s} {'nhân(norm)':>12s} {'nền/sham':>12s} {'tụt-do-NHÂN':>13s} {'dt.sham/nhân':>13s}")
        for k in xs:
            m = (keep == k)
            a = np.nanmean(norm[m]); b = np.nanmean(norm_s[m]); r = np.nanmean(ar_a[m])
            mean_norm.append(a); sd_norm.append(np.nanstd(norm[m])); mean_sh.append(b)
            flag = "" if (k == 1.0 or r >= 0.8) else "  <-- diện tích KHÔNG khớp, so sánh không công bằng"
            print(f"      {1-k:5.2f} {a:12.3f} {b:12.3f} {b - a:13.3f} {r:13.2f}{flag}")
    at_zero = float(np.nanmean(norm[frac_a == 1.0])) if (frac_a == 1.0).any() else float("nan")
    at_zero_s = float(np.nanmean(norm_s[frac_a == 1.0])) if (frac_a == 1.0).any() else float("nan")
    if not args.no_sham:
        ar1 = float(np.nanmean(ar_a[frac_a == 1.0])) if (frac_a == 1.0).any() else float("nan")
        print(f"      gỡ HẾT nhân: count còn {at_zero:.3f} | sham còn {at_zero_s:.3f} -> tụt DO NHÂN = "
              f"{at_zero_s - at_zero:.3f} (dt.sham/nhân={ar1:.2f}{' — sham đặt KHÔNG đủ, chỉ tham khảo' if ar1 < 0.8 else ''})")
        # Wilcoxon chỉ trên hàng khớp diện tích (>=80%): nếu sham gỡ ít hơn thì nó tụt ít là đương nhiên,
        # đưa vào sẽ kết luận "đặc hiệu với nhân" một cách giả.
        from scipy.stats import wilcoxon
        for lab, msh in (("khớp-diện-tích (mọi liều)", (frac_a > 0) & (ar_a >= 0.8)),
                         ("khớp-diện-tích, liều nhỏ ≤30%", (frac_a > 0) & (frac_a <= 0.3) & (ar_a >= 0.8))):
            msh = msh & np.isfinite(norm) & np.isfinite(norm_s)
            if msh.sum() < 10:
                print(f"      [{lab}] chỉ {msh.sum()} mẫu -> BỎ (không đủ để kết luận)")
                continue
            try:
                _, pw = wilcoxon(norm[msh], norm_s[msh])
            except Exception:
                pw = float("nan")
            d = float(np.nanmean(norm_s[msh] - norm[msh]))      # phần tụt DO nhân (đã trừ artifact)
            tot = float(np.nanmean(1 - norm[msh]))              # count tụt khi gỡ nhân
            tots = float(np.nanmean(1 - norm_s[msh]))           # count tụt khi gỡ nền = phần do artifact
            frc = d / tot if tot > 1e-6 else float("nan")       # tỉ lệ cú tụt quy được cho nhân
            # cần cả p nhỏ VÀ effect size đủ lớn. p nhỏ với d≈0 (n lớn) không phải bằng chứng —
            # đó là lỗi p-value-không-effect-size, sẽ tuyên "đặc hiệu" một cách sai.
            ok = (pw < 0.05) and (d >= args.min_effect) and (frc >= 0.5)
            note = " (sham làm TĂNG count -> artifact không gây tụt)" if tots < 0 else ""
            print(f"      [{lab}] n={msh.sum()}: gỡ-nhân tụt {tot:+.3f} | gỡ-nền(artifact) tụt {tots:+.3f} "
                  f"=> DO NHÂN {d:+.3f} = {frc:.0%} cú tụt{note}")
            print(f"      {'':>{len(lab)+3}s} Wilcoxon p={pw:.4g}, ngưỡng effect {args.min_effect} -> "
                  f"{'ĐẶC HIỆU với nhân' if ok else 'KHÔNG đủ (effect nhỏ / phần lớn là artifact inpaint)'}")

    # (d) hồi quy gộp, chỉ để tham chiếu: nó lẫn biến thiên giữa các ảnh nên không dùng để kết luận
    A = np.vstack([rem_a, np.ones_like(rem_a)]).T
    slope, intercept = np.linalg.lstsq(A, pred_a, rcond=None)[0]
    corr = float(np.corrcoef(rem_a, pred_a)[0, 1])
    print(f"  (d) hồi quy gộp pred ~ số nhân còn lại (chỉ để tham chiếu): dốc {slope:.3f}, "
          f"chặn {intercept:+.2f}, corr {corr:.3f} — bị chi phối bởi biến thiên GIỮA ảnh, không phải nhân quả.")
    np.savez(f"{args.out_dir}/occlusion.npz", img=img_a, frac=frac_a, N=N_a, remaining=rem_a,
             pred=pred_a, full=full_a, sham=sham_a, n_sham=nsh_a, area_nuc=anuc_a, area_sham=ash_a,
             slope_within=sl_in, sens_small=sens, slope_pooled=slope, corr=corr)

    plt.figure(figsize=(5, 4))
    plt.errorbar(xs, mean_norm, yerr=sd_norm, fmt="o-", color="darkorange", capsize=3, label="gỡ NHÂN")
    if not args.no_sham:
        plt.plot(xs, mean_sh, "s--", color="steelblue", label="sham: gỡ NỀN cùng diện tích")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="đếm trung thực (y=x)")
    plt.xlabel("Tỉ lệ nhân CÒN LẠI (1 − k)"); plt.ylabel("Count dự đoán / count đầy đủ")
    plt.title(f"Gỡ nhân vs sham\ndốc trong-ảnh={sl_in.mean():.2f}, gỡ-hết={at_zero:.2f} (sham {at_zero_s:.2f})")
    plt.xlim(-0.03, 1.03); plt.ylim(-0.03, 1.1); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{args.out_dir}/doseresp.png", dpi=160); plt.close()

    n_panel = len([f for f in os.listdir(vis_dir) if f.endswith(".png")])
    print(f"\n[xuất] {args.out_dir}/")
    for f in sorted(os.listdir(args.out_dir)):
        p = os.path.join(args.out_dir, f)
        if os.path.isfile(p):
            print(f"    {f:22s} {os.path.getsize(p)/1024:8.1f} KB")
    print(f"    samples/               {n_panel} panel ảnh (gốc | gỡ 30% | gỡ hết | sham)")
    if args.zip:
        import shutil
        z = shutil.make_archive(args.out_dir, "zip", args.out_dir)
        print(f"[zip] {z} ({os.path.getsize(z)/1e6:.1f} MB) — tải 1 lần là đủ")
    print("ĐỌC (): scatter bám chéo + R² từng cơ quan > 0 + bias≈0 = đếm đúng, không thiên vị hệ thống.")
    print("ĐỌC (): CHỈ được kết luận nhân-quả nếu (a) dốc TRONG-ẢNH ≈1 VÀ (b) Δ/nhân ≈1 VÀ (c) sham "
          "tụt ÍT hơn rõ rệt (p<0.05). Nếu 'tụt-do-NHÂN' nhỏ -> phần lớn là ARTIFACT inpaint, KHÔNG "
          "phải bằng chứng; kết quả ở mức bất định.")


if __name__ == "__main__":
    main()
