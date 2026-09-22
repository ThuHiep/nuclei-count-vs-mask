from __future__ import annotations
import argparse, os, pickle, sys, time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(_HERE, "lib"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_io import build_index, find_root, IMG_SIZE, _load_mask, assign_kfold  # noqa: E402
from losses import r2_loss, count_from_density  # noqa: E402


class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

    def forward(self, x): return self.net(x)


# ===================== Mô hình: đầu mật độ + đầu log sigma =====================
class _UpBlock(nn.Module):
    """Decoder block cho backbone timm: upsample x về size skip, concat, DoubleConv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class DensitySigmaUNet(nn.Module):
    """Encoder-decoder -> (density_map>=0, log_sigma scalar/ảnh).

    backbone:
      'efficientnet_lite0' (MẶC ĐỊNH, cấu hình của paper) — 3.62M params, 0.98 GMACs @256.
      'tinyunet' — U-Net thuan DoubleConv, ch=32 => 1,94M tham so. Khong dung trong bai;
          giu lai de nap cac diem kiem tra cu.
      '<ten timm> (vd 'efficientnet_lite0', 'convnext_atto') — bo ma hoa timm (features_only)
          + bo giai ma U-Net + hai dau ra (mat do, sigma). Day la cac cau hinh dung trong bai.

    sigma_mode:
      'poisson' (mặc định) — σ = √(max(μ,1)) · exp(log_s); anchor Poisson + head học dispersion.
      'nb' — σ = √(μ+α·μ²) Negative-Binomial (ablation A2). 'raw' — σ = exp(log_s) (ablation A2)."""
    def __init__(self, ch=32, sigma_mode="poisson", backbone="efficientnet_lite0", head_k=1, head_d=1):
        super().__init__()
        self.sigma_mode = sigma_mode
        self.backbone = backbone
        if backbone == "tinyunet":
            self.d1 = DoubleConv(3, ch);          self.p1 = nn.MaxPool2d(2)
            self.d2 = DoubleConv(ch, ch * 2);     self.p2 = nn.MaxPool2d(2)
            self.d3 = DoubleConv(ch * 2, ch * 4); self.p3 = nn.MaxPool2d(2)
            self.bott = DoubleConv(ch * 4, ch * 8)
            self.u3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2); self.c3 = DoubleConv(ch * 8, ch * 4)
            self.u2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2); self.c2 = DoubleConv(ch * 4, ch * 2)
            self.u1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2);     self.c1 = DoubleConv(ch * 2, ch)
            sig_in = ch * 8
        else:
            import timm
            self.enc = timm.create_model(backbone, features_only=True, pretrained=True, in_chans=3)
            chs = list(self.enc.feature_info.channels())   # low->high stride
            rev = list(reversed(chs))                       # deep->shallow
            self.ups = nn.ModuleList()
            prev = rev[0]
            for skip in rev[1:]:
                self.ups.append(_UpBlock(prev, skip, skip)); prev = skip
            self.dec_out = nn.Sequential(nn.Conv2d(prev, ch, 3, padding=1), nn.ReLU(inplace=True))
            sig_in = chs[-1]
        _pad = head_d * (head_k // 2)              # head_k=1 -> 1x1 (mặc định, như CSRNet); >1 -> gộp cục bộ lúc readout
        self.dens = nn.Conv2d(ch, 1, head_k, padding=_pad, dilation=head_d)   # -> ReLU = density >=0
        # head log_s: pool feature sâu nhất -> MLP -> scalar (dispersion/ảnh)
        self.sig = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(sig_in, ch), nn.ReLU(inplace=True), nn.Linear(ch, 1))
        _binit = {"poisson": 0.0, "nb": -2.0}.get(sigma_mode, 2.7)
        nn.init.constant_(self.sig[-1].bias, _binit)

    def _features(self, x):
        """Trả (y: feature ch-kênh ở FULL res cho density head, fs: feature sâu nhất cho σ head)."""
        if self.backbone == "tinyunet":
            x1 = self.d1(x); x2 = self.d2(self.p1(x1)); x3 = self.d3(self.p2(x2))
            xb = self.bott(self.p3(x3))
            y = self.c3(torch.cat([self.u3(xb), x3], 1))
            y = self.c2(torch.cat([self.u2(y), x2], 1))
            y = self.c1(torch.cat([self.u1(y), x1], 1))
            return y, xb
        feats = list(reversed(self.enc(x)))                 # deep->shallow
        y = feats[0]
        for i, up in enumerate(self.ups):
            y = up(y, feats[i + 1])
        y = F.interpolate(self.dec_out(y), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y, feats[0]

    def forward(self, x):
        y, fs = self._features(x)
        density = F.relu(self.dens(y))             # (B,1,H,W) >=0
        log_s = self.sig(fs).squeeze(1)            # (B,) dispersion thô
        if self.sigma_mode == "poisson":
            mu = density.sum(dim=(1, 2, 3)).detach()          # count anchor (detach: σ mượn độ lớn, không kéo μ)
            log_sigma = 0.5 * torch.log(torch.clamp(mu, min=1.0)) + torch.clamp(log_s, -2.0, 2.0)
        elif self.sigma_mode == "nb":
            # Negative-Binomial variance: Var = μ + α·μ² (α=overdispersion học được; α0 = Poisson).
            mu = density.sum(dim=(1, 2, 3)).detach()
            alpha = torch.exp(torch.clamp(log_s, -6.0, 2.0))  # log_s = log(α)
            var = torch.clamp(mu, min=1.0) + alpha * mu ** 2
            log_sigma = 0.5 * torch.log(torch.clamp(var, min=1.0))
        else:  # 'raw' = Gaussian heteroscedastic thuần: σ=exp(log_s), không neo-mean
            log_sigma = log_s
        return density, log_sigma

    @torch.no_grad()
    def pooled_feat(self, x):
        """Đặc trưng sâu/ảnh = global-avg-pool feature sâu nhất (cho baseline R2CCP/FFCP, leak-free)."""
        _, fs = self._features(x)
        return fs.mean(dim=(2, 3))


# ===================== Dung du lieu: anh + ban do mat do muc tieu =====================
@torch.no_grad()
def build_pannuke_density(root, folds, device, cache):
    """PanNuke -> list {img, density, gt, organ, fold}, luu cache pickle.

    Nhan dem `gt` lay tu so instance da chu thich. Ban do mat do dung tu mat na instance:
    moi nhan gop khoi luong 1.0 trai deu tren cac diem anh cua no, gop chung 5 lop.
    `organ` la loai mo (19 loai), dung cho phan tich theo tung co quan.
    """
    if os.path.exists(cache):
        print(f"[data] doc cache {cache}")
        return pickle.load(open(cache, "rb"))
    from pannuke_loader import PanNukeFold
    data, t0, n_done = [], time.time(), 0
    for fold in folds:
        pf = PanNukeFold(root, fold)
        for i in range(len(pf)):
            s = pf[i]
            img = s["image"]                          # (256,256,3) uint8
            gt = int(s["counts"].sum())
            if s["masks"] is None:
                raise RuntimeError("Can instance mask (masks.npy) de dung ban do mat do.")
            dens = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
            for k in range(5):
                lab = s["masks"][k]
                for iid in np.unique(lab):
                    if iid == 0:
                        continue
                    mr = np.asarray(Image.fromarray((lab == iid).astype(np.uint8)).resize(
                        (IMG_SIZE, IMG_SIZE), Image.NEAREST)).astype(bool)
                    a = int(mr.sum())
                    if a > 0:
                        dens[mr] += 1.0 / a
            img_r = np.asarray(Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
            data.append({"img": img_r.astype(np.uint8), "density": dens,
                         "gt": float(gt), "organ": s["tissue"], "fold": int(fold)})
            n_done += 1
            if n_done % 200 == 0:
                print(f"[data] {n_done} anh, {(time.time()-t0)/n_done:.2f}s/anh")
    pickle.dump(data, open(cache, "wb"))
    print(f"[data] da luu {cache} ({len(data)} anh)")
    return data


@torch.no_grad()
def build_nuinsseg_density(samples, device, cache):
    """NuInsSeg -> list {img, density, gt, organ}, cung quy uoc voi build_pannuke_density.

    Anh goc 512x512 duoc thu nho ve IMG_SIZE; mat na tung instance thu nho bang lan can gan nhat.
    """
    if os.path.exists(cache):
        print(f"[data] doc cache {cache}")
        return pickle.load(open(cache, "rb"))
    data, t0 = [], time.time()
    for k, s in enumerate(samples):
        img = np.asarray(Image.open(s["image"]).convert("RGB"))
        m = _load_mask(s["mask"])
        gt = int(len(np.unique(m)) - (1 if (m == 0).any() else 0))
        dens = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
        for iid in np.unique(m):
            if iid == 0:
                continue
            mr = np.asarray(Image.fromarray((m == iid).astype(np.uint8)).resize(
                (IMG_SIZE, IMG_SIZE), Image.NEAREST)).astype(bool)
            a = int(mr.sum())
            if a > 0:
                dens[mr] += 1.0 / a
        img_r = np.asarray(Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        data.append({"img": img_r.astype(np.uint8), "density": dens,
                     "gt": float(gt), "organ": s["organ"]})
        if (k + 1) % 100 == 0:
            print(f"[data] {k+1}/{len(samples)} {(time.time()-t0)/(k+1):.2f}s/anh")
    pickle.dump(data, open(cache, "wb"))
    print(f"[data] da luu {cache}")
    return data


# ===================== Phase B: train =====================
def train(data, device, epochs, ch, lr, train_idx, w_density, w_count, w_nll, beta, bs,
          detach_mu=False, sigma_mode="poisson", backbone="efficientnet_lite0", head_k=1, head_d=1,
          preload=False):
    model = DensitySigmaUNet(ch, sigma_mode=sigma_mode, backbone=backbone, head_k=head_k, head_d=head_d).to(device)
    print(f"[B] DensitySigmaUNet backbone={backbone} ch={ch} params={sum(p.numel() for p in model.parameters())/1e6:.3f}M "
          f"w=(dens {w_density}, count {w_count}, nll {w_nll}) beta={beta} detach_mu={detach_mu} sigma_mode={sigma_mode}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    # preload (opt-in): đẩy toàn tập train lên GPU 1 lần -> index trên GPU, bỏ stack numpy + copy H2D mỗi batch
    # (nút cổ chai làm epoch chậm ~10-20x trên dataset lớn). OOM nếu tập quá lớn -> chỉ bật khi vừa VRAM.
    if preload:
        _imgs = torch.from_numpy(np.stack([data[j]["img"] for j in train_idx])).to(device)     # (N,H,W,3) uint8
        _dens = torch.from_numpy(np.stack([data[j]["density"] for j in train_idx])).to(device)  # (N,H,W) f32
        _gt = torch.tensor([data[j]["gt"] for j in train_idx], device=device, dtype=torch.float32)
        _pos = {j: k for k, j in enumerate(train_idx)}
        print(f"[B] preload GPU: imgs {tuple(_imgs.shape)} dens {tuple(_dens.shape)} "
              f"(~{(_imgs.numel()+_dens.numel()*4)/1e9:.1f}GB) -> epoch không còn nghẽn CPU")
    for ep in range(epochs):
        np.random.shuffle(train_idx)
        logs = {"loss": [], "L_density": [], "L_count": [], "L_nll": [], "sigma": []}
        for i in range(0, len(train_idx), bs):
            idxs = train_idx[i:i + bs]
            if preload:
                rows = torch.as_tensor([_pos[j] for j in idxs], device=device)
                imgs = _imgs[rows].permute(0, 3, 1, 2).float() / 255.0
                dT = _dens[rows][:, None]
                gt = _gt[rows]
            else:
                imgs = torch.from_numpy(
                    np.stack([data[j]["img"] for j in idxs]).astype(np.float32) / 255.0
                ).permute(0, 3, 1, 2).to(device)
                dT = torch.from_numpy(
                    np.stack([data[j]["density"] for j in idxs]))[:, None].to(device)
                gt = torch.tensor([data[j]["gt"] for j in idxs], device=device, dtype=torch.float32)
            dens_S, log_sigma = model(imgs)
            out = r2_loss(dens_S, dT, log_sigma, gt, w_density, w_count, w_nll, beta, detach_mu)
            opt.zero_grad(); out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # chống phân kỳ hiếm (β-NLL spike)
            opt.step()
            for key in ("loss", "L_density", "L_count", "L_nll"):
                logs[key].append(float(out[key]))
            logs["sigma"].append(float(out["sigma"].mean()))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[B] ep {ep+1}/{epochs} loss={np.mean(logs['loss']):.4f} "
                  f"Ldens={np.mean(logs['L_density']):.4f} Lcount={np.mean(logs['L_count']):.3f} "
                  f"Lnll={np.mean(logs['L_nll']):.3f} sigma~{np.mean(logs['sigma']):.2f}")
    return model


# ===================== Phase C: inference -> (mu, sigma) =====================
@torch.no_grad()
def predict_r2(model, data, device, dump_feat=False):
    model.eval()
    preds, gts, organs = [], [], []
    for d in data:
        img = torch.from_numpy(d["img"].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        dens_S, log_sigma = model(img)
        mu = float(count_from_density(dens_S)[0])
        sigma = float(torch.exp(log_sigma)[0])
        p = {"mu": mu, "sigma": sigma}
        if dump_feat:
            p["feat"] = model.pooled_feat(img)[0].cpu().numpy().astype(np.float32)  # (ch*8,) leak-free
        preds.append(p)
        gts.append([d["gt"]]); organs.append(d["organ"])
    return {"preds": preds, "gts": gts, "organs": organs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--backbone", default="efficientnet_lite0",
                    help="efficientnet_lite0 (mặc định = cấu hình paper, 3.62M) HOẶC tên timm khac / 'tinyunet' (model CU 1.9M) "
                         "-> encoder hiện đại + U-Net decoder, GIỮ Poisson-σ + detach_mu. Bước 1 redesign accuracy.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_density", type=float, default=1.0)
    ap.add_argument("--w_count", type=float, default=0.01, help="L_count là |mu-GT| (thang chục) -> trọng số nhỏ")
    ap.add_argument("--w_nll", type=float, default=0.01, help="L_nll thang lớn -> trọng số nhỏ để cân với density MSE")
    ap.add_argument("--beta", type=float, default=0.5, help="beta-NLL (Seitzer 2022)")
    ap.add_argument("--sigma_mode", choices=["poisson", "raw", "nb"], default="poisson",
                    help="poisson: σ=√(max(μ,1))·exp(log_s) (mặc định); raw: σ=exp(log_s) (Gaussian-hetero baseline); "
                         "nb: σ=√(μ+α·μ²) Negative-Binomial (baseline A2 overdispersion tường minh)")
    ap.add_argument("--detach_mu", action="store_true",
                    help="tách mu khỏi NLL (NLL chỉ dạy sigma) — sửa NLL-coupling làm hỏng MAE")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--dataset", choices=["nuinsseg", "pannuke"], default="nuinsseg")
    ap.add_argument("--pannuke_root", default="data/pannuke")
    ap.add_argument("--pannuke_folds", default="1,2,3", help="fold PanNuke dùng, vd '1,2,3' hoặc '3'")
    ap.add_argument("--test_fold", type=int, default=None,
                    help="PanNuke: HELD-OUT fold để test (leak-free). Train trên các fold còn lại, "
                         "predict CHỈ trên test_fold. Bỏ trống -> train+predict toàn bộ (chỉ để debug).")
    ap.add_argument("--kfold", type=int, default=None,
                    help="NuInsSeg: cross-fitting K-fold (leak-free). Train K-1 fold -> predict fold held-out, "
                         "ghép mọi ảnh (mỗi ảnh dự đoán bởi model KHÔNG train nó). Vd 5.")
    ap.add_argument("--exclude_tissue", default=None,
                    help="PanNuke: loai tissue chua chuoi nay (khong phan biet hoa thuong, cach nhau ',').")
    ap.add_argument("--dump_feat", action="store_true",
                    help="lưu thêm đặc trưng sâu/ảnh (pooled bottleneck) vào pkl -> cho baseline R2CCP/FFCP")
    ap.add_argument("--cache", default=None, help="mặc định tự đặt theo dataset")
    ap.add_argument("--out", default="work/preds_nuinsseg.pkl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} dataset={args.dataset}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.dataset == "pannuke":
        folds = [int(x) for x in args.pannuke_folds.split(",")]
        fstr = "".join(str(x) for x in sorted(folds))   # cache theo tập fold -> không đụng cache cũ
        cache = args.cache or f"work/gt_density_pannuke_f{fstr}.pkl"
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        data = build_pannuke_density(args.pannuke_root, folds, device, cache)
    else:
        cache = args.cache or "work/gt_density_nuinsseg.pkl"
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        if os.path.exists(cache):
            samples = None   # cache tự chứa img/gt/organ -> không cần raw NuInsSeg
            print(f"[A] cache có sẵn -> bỏ qua build_index (không cần raw data)")
        else:
            samples = build_index(find_root())
            print(f"indexed {len(samples)} pairs")
        data = build_nuinsseg_density(samples, device, cache)

    # --- loai bo mot so loai mo neu duoc yeu cau ---
    if args.exclude_tissue:
        ex = [t.strip().lower() for t in args.exclude_tissue.split(",") if t.strip()]
        before = len(data)
        data = [d for d in data if not any(e in str(d["organ"]).lower() for e in ex)]
        print(f"[EXCLUDE] bỏ tissue chứa {ex}: {before} -> {len(data)} ảnh (loại {before-len(data)})")

    # --- cross-fitting K-fold (leak-free cho NuInsSeg: mỗi ảnh dự đoán bởi model không train nó) ---
    if args.kfold and args.kfold > 1:
        N = len(data)
        fold_of = assign_kfold([d["organ"] for d in data], args.kfold, args.seed)
        all_p = [None] * N; all_g = [None] * N; all_o = [None] * N
        for f in range(args.kfold):
            tr = [i for i in range(N) if fold_of[i] != f]
            te = [i for i in range(N) if fold_of[i] == f]
            print(f"[CV] fold {f+1}/{args.kfold}: train {len(tr)} | held-out predict {len(te)}")
            m = train(data, device, args.epochs, args.ch, args.lr, tr,
                      args.w_density, args.w_count, args.w_nll, args.beta, args.bs,
                      args.detach_mu, args.sigma_mode, args.backbone)
            of = predict_r2(m, [data[i] for i in te], device, dump_feat=args.dump_feat)
            for k, i in enumerate(te):
                all_p[i] = of["preds"][k]; all_g[i] = of["gts"][k]; all_o[i] = of["organs"][k]
        out = {"preds": all_p, "gts": all_g, "organs": all_o}
        print(f"[CV] ghép {N} dự đoán leak-free ({args.kfold}-fold cross-fitting)")
    else:
        # --- tách train/test theo fold (PanNuke) hoặc train-all (debug) ---
        if args.dataset == "pannuke" and args.test_fold is not None:
            train_idx = [i for i, d in enumerate(data) if d["fold"] != args.test_fold]
            test_data = [d for d in data if d["fold"] == args.test_fold]
            assert train_idx and test_data, f"test_fold={args.test_fold} không tách được (train={len(train_idx)}, test={len(test_data)})"
            n_tr_folds = sorted({d["fold"] for d in data if d["fold"] != args.test_fold})
            print(f"[SPLIT] train folds={n_tr_folds} ({len(train_idx)} imgs) | "
                  f"TEST fold={args.test_fold} ({len(test_data)} imgs) — leak-free")
        else:
            train_idx = list(range(len(data)))
            test_data = data
            print("[SPLIT] WARN: train+predict TOÀN BỘ (LEAK). Dùng --kfold (NuInsSeg) / --test_fold (PanNuke).")
        model = train(data, device, args.epochs, args.ch, args.lr, train_idx,
                      args.w_density, args.w_count, args.w_nll, args.beta, args.bs,
                      args.detach_mu, args.sigma_mode, args.backbone)
        out = predict_r2(model, test_data, device, dump_feat=args.dump_feat)
    pickle.dump(out, open(args.out, "wb"))
    mu = np.array([p["mu"] for p in out["preds"]])
    sg = np.array([p["sigma"] for p in out["preds"]])
    gtv = np.array([g[0] for g in out["gts"]])
    print(f"[C] saved {args.out} | MAE={np.abs(mu-gtv).mean():.2f} "
          f"| sigma mean={sg.mean():.2f} std={sg.std():.2f} (std>0 => heteroscedastic)")


if __name__ == "__main__":
    main()
