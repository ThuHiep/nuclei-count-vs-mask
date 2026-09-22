#!/usr/bin/env python3
import argparse, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "lib"), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
from dsunet import build_pannuke_density, train
from eval_metrics import eval_r2_mae
from losses import count_from_density


@torch.no_grad()
def _per_image_pred(model, data, idx, dev):
    model.eval(); pr = []
    for i in idx:
        x = torch.from_numpy(data[i]["img"].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        pr.append(float(count_from_density(model(x)[0])[0]))
    return np.array(pr)


def compare(per_img, gt_te, seeds, args, rows=None):
    """So 2 arm: Δ R² + per-image Wilcoxon gộp-seed. Tách riêng để dùng được cho cả run có train
    lẫn chế độ gộp-thuần (--arms none)."""
    have = [n for n in ("count-only", "GT-density (mask)") if len(per_img.get(n, []))]
    if len(have) < 2:
        print(f"\n  [1 arm] chỉ có '{have[0] if have else 'không'}' -> BỎ QUA Δ và Wilcoxon. "
              f"Dùng --dump_preds rồi '--arms none --load_preds a.npz,b.npz' để gộp với arm kia.")
        return
    if rows and len(rows) == 2:
        print(f"\n  Δ(count-only − GT-density) = {rows[0][1] - rows[1][1]:+.4f} R² (mean) | "
              f"{rows[0][5] - rows[1][5]:+.4f} (median, bền hơn khi có seed sập)")
    ec = np.mean([np.abs(gt_te - p) for p in per_img["count-only"]], axis=0)
    em = np.mean([np.abs(gt_te - p) for p in per_img["GT-density (mask)"]], axis=0)
    from scipy.stats import wilcoxon
    try:
        _, pw = wilcoxon(ec, em)
    except Exception as e:
        pw = float("nan"); print(f"[wilcoxon lỗi] {e}")
    dm = float((ec - em).mean())
    ket_luan = ("không phân biệt được (p>0.05)" if pw > 0.05 else
               ("count-only THẮNG (p<0.05, lỗi ít hơn)" if dm < 0 else "count-only THUA (p<0.05)"))
    print(f"  per-image |err| (gộp {len(per_img['count-only'])} lần train count + "
          f"{len(per_img['GT-density (mask)'])} mask, n={len(gt_te)}): count-only {ec.mean():.2f} vs "
          f"mask {em.mean():.2f} | Wilcoxon p={pw:.4g} -> {ket_luan}")
    if rows and any(r[6] for r in rows):
        print("  có seed sập trong nhóm này nên p ở trên không phản ánh giám sát.")
    elif pw > 0.05 or dm < 0:
        print("  count-only không kém mask.")
    else:
        print(f"  mask nhỉnh hơn {dm:+.3f} nhân/ảnh và khác biệt đạt ý nghĩa thống kê.")
    # p > 0.05 chỉ là không bác bỏ được "khác nhau"; muốn kết luận tương đương thì phải xem TOST.
    try:
        from equivalence import report as tost_report
        tost_report(ec, em, gt=gt_te, delta=args.delta, delta_frac=args.delta_frac)
    except Exception as e:
        print(f"  [TOST lỗi] {e}")


def load_npz_any(path):
    """Nạp .npz kể cả khi nó đã bị GIẢI NÉN thành thư mục. File .npz là một zip, nên trình upload dataset
    (hoặc Safari khi tải về) có thể bung nó ra thành <name>/{key1.npy, key2.npy, ...}. Gặp thư mục thì
    nạp từng .npy thành viên -> trả về dict có cùng giao diện .files/[] như np.load."""
    if os.path.isdir(path):
        d = {f[:-4]: np.load(os.path.join(path, f), allow_pickle=True)
             for f in sorted(os.listdir(path)) if f.endswith(".npy")}
        assert d, f"{path} là thư mục nhưng không có .npy nào bên trong"
        print(f"[load_preds] '{os.path.basename(path)}' là THƯ MỤC (npz đã bị giải nén) -> nạp {len(d)} mảng .npy")
        class _D(dict):
            files = property(lambda self: list(self.keys()))
        return _D(d)
    return np.load(path, allow_pickle=True)


def stats_only(args):
    """--arms none: chỉ nạp .npz rồi thống kê. Không train, không đọc dataset -> chạy vài giây ở đâu cũng được."""
    per_img, gt_te, r2s = {}, None, {}
    for f in [x.strip() for x in args.load_preds.split(",") if x.strip()]:
        z = load_npz_any(f)
        if gt_te is None:
            gt_te = z["gt"]
        assert np.allclose(gt_te, z["gt"]), f"{f}: GT khác -> khác test set, KHÔNG được gộp"
        if "pred" in z.files and not any(k.startswith("pred__") for k in z.files):
            # định dạng deletion_doseresp.py (preds.npz: pred/gt/organ/R2) = count-only, 1 seed/file.
            # Cấu hình huấn luyện giống hệt nhánh count của script này (60 epoch, ch32, lr 1e-3,
            # bs 16, w=(0,1,0.01,0.5), detach_mu, sigma poisson) và cùng fold kiểm tra nên gộp được.
            per_img.setdefault("count-only", []).append(np.asarray(z["pred"]))
            r2s.setdefault("count-only", []).append(float(z["R2"]) if "R2" in z.files else float("nan"))
            print(f"[load_preds] {f}: deletion_doseresp preds.npz -> count-only (1 seed, R²={float(z['R2']):+.4f})")
            continue
        for k in z.files:
            if k.startswith("pred__"):
                per_img.setdefault(k[6:].replace("_", " "), []).extend(list(z[k]))
            elif k.startswith("r2__"):
                r2s.setdefault(k[4:].replace("_", " "), []).extend(list(np.atleast_1d(z[k])))
        print(f"[load_preds] {f}: arm {[k[6:] for k in z.files if k.startswith('pred__')]}")
    print(f"\n=== SUPERVISION head-to-head (gộp từ .npz, n={len(gt_te)} ảnh) ===")
    rows = []
    for nm, v in r2s.items():
        v = np.array(v, float); nc = int((v < args.collapse_thr).sum())
        rows.append((nm, v.mean(), v.std(), np.nan, np.nan, float(np.median(v)), nc, list(v)))
        print(f"  {nm:20s} R² {v.mean():+.4f}±{v.std():.3f}  median {np.median(v):+.4f}  SẬP {nc}/{len(v)}"
              f"  | per-seed {np.round(v, 4).tolist()}")
    rows.sort(key=lambda r: {"count-only": 0, "GT-density (mask)": 1}.get(r[0], 9))
    compare(per_img, gt_te, None, args, rows if len(rows) == 2 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pannuke_root", default="", help="bắt buộc, TRỪ khi --arms none (chỉ gộp .npz).")
    ap.add_argument("--train_folds", default="1,2")
    ap.add_argument("--test_fold", default="3")
    ap.add_argument("--backbone", default="efficientnet_lite0")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)   # density cần lâu hơn để hội tụ; count-only đã bão hoà ~40
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--bs", type=int, default=16,
                    help="batch size. Tăng (48/64) để GPU no hơn (T4 15GB thừa) -> nhanh hơn nhiều/epoch.")
    ap.add_argument("--preload", action="store_true",
                    help="đẩy toàn tập train lên GPU 1 lần (bỏ nghẽn CPU stack/copy mỗi batch) -> nhanh ~5-15x. "
                         "PanNuke fold12 ~2.4GB GPU, vừa T4. TẮT nếu OOM.")
    ap.add_argument("--w_density", type=float, default=6553.6,
                    help="w_d = H*W*w_c = 65536*0.1 (lap luan don vi)")
    ap.add_argument("--w_count_mask", type=float, default=0.1,
                    help="neo count cho mask (0.1 ổn định; 0.01 cũ có thể sập theo seed)")
    ap.add_argument("--arms", default="count,mask",
                    help="nhánh nào chạy: 'count', 'mask', hoặc 'count,mask'. Tách ra để khỏi chạy lại nhánh đã xong.")
    ap.add_argument("--dump_preds", default="",
                    help="lưu preds per-image từng arm/seed vào .npz. LUÔN BẬT cho run đắt: mất preds là mất "
                         "cả run (không tính lại Wilcoxon được). Gộp 2 notebook bằng --load_preds.")
    ap.add_argument("--load_preds", default="",
                    help="một hoặc NHIỀU .npz (phân tách bằng dấu phẩy) từ --dump_preds -> gộp lại. "
                         "Dùng với --arms none để CHỈ gộp+thống kê: không train, không cần dataset, chạy "
                         "được trên máy thường trong vài giây.")
    ap.add_argument("--collapse_thr", type=float, default=0.5,
                    help="R² dưới ngưỡng này = seed SẬP tối ưu hoá (density-MSE bị background chi phối). "
                         "Báo riêng số seed sập + median, vì mean bị 1 seed sập kéo lệch vô nghĩa.")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Adam LR; efflite0/CNN dùng 1e-3. Backbone transformer (pvt/swin) có thể cần "
                         "hạ (3e-4/1e-4) nếu 1e-3 phá feature pretrained -> sập hằng số.")
    ap.add_argument("--ckpt_dir", default="",
                    help="lưu state_dict mỗi arm/seed vào đây (dat ngoai kho ma nguon). Trước đây chỉ "
                         "dump_preds nên KHÔNG còn model nhánh mask để vẽ lại density map.")
    ap.add_argument("--work_dir", default="work",
                    help="nơi lưu cache GT-density. dat ngoai kho ma nguon (dat ngoai kho ma nguon) "
                         "để 'rm -rf <repo>' mỗi lần clone KHÔNG xóa cache -> khỏi build lại (rất chậm).")
    ap.add_argument("--delta", type=float, default=None,
                    help="TOST: biên tương đương (đơn vị NHÂN). Mặc định None -> suy từ --delta_frac.")
    ap.add_argument("--delta_frac", type=float, default=0.05,
                    help="TOST: biên = delta_frac * số nhân trung bình của tập kiểm tra (mặc định 5%%).")
    ap.add_argument("--deterministic", action="store_true",
                    help="bật cuDNN tất định — CHẬM hơn nhiều với efflite0. PanNuke lớn+ổn định KHÔNG cần "
                         "-> mặc định TẮT (benchmark=True) cho nhanh.")
    args = ap.parse_args()
    if args.arms.strip().lower() in ("none", ""):        # gộp-thuần: không train, không cần dataset
        assert args.load_preds, "--arms none cần --load_preds <npz>[,<npz>...]"
        return stats_only(args)
    assert args.pannuke_root, "--pannuke_root bắt buộc (trừ khi --arms none)"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    try:                                       # work_dir co the chi doc
        os.makedirs(args.work_dir, exist_ok=True)   # nơi lưu cache mật độ
    except OSError as e:
        print(f"[work_dir] không tạo được ({e}) — chỉ đọc cache")
    # kiem quyen ghi truoc khi train
    for _d in (args.ckpt_dir, os.path.dirname(os.path.abspath(args.dump_preds)) if args.dump_preds else ""):
        if _d:
            os.makedirs(_d, exist_ok=True)
            _p = os.path.join(_d, ".probe"); open(_p, "w").close(); os.remove(_p)

    tr_folds = [int(x) for x in args.train_folds.split(",")]
    fstr = "".join(str(x) for x in sorted(tr_folds))
    train_data = build_pannuke_density(args.pannuke_root, tr_folds, dev,
                                       f"{args.work_dir}/gt_density_pannuke_f{fstr}.pkl")
    test_data = build_pannuke_density(args.pannuke_root, [int(args.test_fold)], dev,
                                      f"{args.work_dir}/gt_density_pannuke_f{args.test_fold}.pkl")
    data = list(train_data) + list(test_data)
    tr = list(range(len(train_data))); te = list(range(len(train_data), len(data)))
    print(f"[data] train fold{tr_folds}={len(tr)} / test fold{args.test_fold}={len(te)} "
          f"(CÙNG data, cùng efflite0 ch{args.ch}; chỉ đổi loss)")

    seeds = [int(s) for s in args.seeds.split(",")]
    # GT-density cần neo count nhỏ (0.01) để train được — density MSE thuần bị background(0) chi phối
    # collapse về ~0 nếu w_count=0 (đúng recipe density-counting chuẩn; số cũ NuInsSeg 0.881 dùng vậy).
    _ARM = {"count": ("count-only", 0.0, 1.0),
            "mask": ("GT-density (mask)", args.w_density, args.w_count_mask)}
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    assert all(a in _ARM for a in arms), f"--arms chỉ nhận count/mask, nhận được {arms}"
    configs = [_ARM[a] for a in arms]
    rows = []
    per_img = {name: [] for name, _, _ in configs}   # preds mỗi seed (fold3 cố định) -> Wilcoxon gộp-seed
    gt_te = np.array([data[i]["gt"] for i in te])
    if args.load_preds:                              # gộp arm đã chạy ở notebook khác
        z = load_npz_any(args.load_preds)
        for k in z.files:
            if k.startswith("pred__"):
                nm = str(k[6:]).replace("_", " ")
                per_img.setdefault(nm, []).extend(list(z[k]))
                print(f"[load_preds] nạp '{nm}': {len(z[k])} seed × {len(z[k][0])} ảnh từ {args.load_preds}")
    for key, (name, wd, wc) in zip(arms, configs):
        r2s, maes = [], []
        for sd in seeds:
            np.random.seed(sd); torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
            if args.deterministic:
                # cuDNN tất định: mỗi seed tái lập chính xác nhưng chậm (efflite0 depthwise). Chỉ bật khi cần.
                torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True, warn_only=True)
            else:
                torch.backends.cudnn.benchmark = True   # nhanh; PanNuke lớn+ổn định nên không cần tất định
            # train(data, device, epochs, ch, lr, train_idx, w_density, w_count, w_nll, beta, bs, detach_mu, sigma_mode, backbone)
            ck = f"{args.ckpt_dir}/{key}_s{sd}.pt" if args.ckpt_dir else None
            if ck and os.path.exists(ck):        # resume: seed da chay xong thi khoi train lai
                from dsunet import DensitySigmaUNet
                model = DensitySigmaUNet(args.ch, sigma_mode="poisson", backbone=args.backbone).to(dev)
                model.load_state_dict(torch.load(ck, map_location=dev))
            else:
                model = train(data, dev, args.epochs, args.ch, args.lr, tr,
                              wd, wc, 0.01, 0.5, args.bs, True, "poisson", args.backbone,
                              preload=args.preload)
                if ck:
                    os.makedirs(args.ckpt_dir, exist_ok=True)
                    torch.save(model.state_dict(), ck)
            r2, mae = eval_r2_mae(model, data, te, dev)
            r2s.append(r2); maes.append(mae)
            per_img[name].append(_per_image_pred(model, data, te, dev))
            print(f"    [{name:16s}] seed {sd}: R² {r2:+.4f}  MAE {mae:.2f}")   # per-seed để bắt collapse
            if args.dump_preds:                  # luu ngay: 1 seed ~1.5h, dung de mat khi phien dut
                np.savez(args.dump_preds, gt=gt_te, seeds=np.array(seeds),
                         **{f"pred__{n2.replace(' ', '_')}": np.array(v)
                            for n2, v in per_img.items() if len(v)},
                         **{f"r2__{name.replace(' ', '_')}": np.array(r2s)})
        nc = int(np.sum(np.array(r2s) < args.collapse_thr))
        rows.append((name, np.mean(r2s), np.std(r2s), np.mean(maes), np.std(maes),
                     float(np.median(r2s)), nc, list(r2s)))
        print(f"  {name:20s} R² {np.mean(r2s):+.4f}±{np.std(r2s):.3f}  MAE {np.mean(maes):.3f}±{np.std(maes):.3f}"
              f"  | median R² {np.median(r2s):+.4f}, SẬP {nc}/{len(r2s)} seed (R²<{args.collapse_thr})")

    if args.dump_preds:
        # lưu dự đoán để tính lại thống kê mà không phải huấn luyện lại, và để gộp các nhánh
        # chạy ở phiên khác qua --load_preds.
        d = os.path.dirname(os.path.abspath(args.dump_preds))
        os.makedirs(d, exist_ok=True)
        np.savez(args.dump_preds, gt=gt_te, seeds=np.array(seeds),
                 **{f"pred__{n.replace(' ', '_')}": np.array(v) for n, v in per_img.items() if len(v)},
                 **{f"r2__{n.replace(' ', '_')}": np.array(r[7]) for n, r in zip([x[0] for x in rows], rows)})
        print(f"[dump_preds] đã lưu {args.dump_preds}")

    print(f"\n=== SUPERVISION head-to-head (CÙNG {args.backbone} ch{args.ch}, lr{args.lr:g}, {len(seeds)} seed, "
          f"PanNuke fold{args.test_fold}, w_count_mask={args.w_count_mask:g}) ===")
    print(f"  {'supervision':20s} {'R² mean ':>16s} {'R² median':>11s} {'sập':>6s} {'MAE ':>14s}")
    for name, r2m, r2sd, mm, ms, r2med, nc, _ in rows:
        print(f"  {name:20s} {r2m:+.4f}±{r2sd:.3f} {r2med:+11.4f} {nc:4d}/{len(seeds)}   {mm:6.2f}±{ms:.2f}")
    bad = [r[0] for r in rows if r[6]]
    if bad:
        print(f"  SẬP ở: {', '.join(bad)} -> mean VÔ NGHĨA, đọc median. Sập = thất bại TỐI ƯU HOÁ, KHÔNG "
              "phải giới hạn của loại giám sát; nên sửa cấu hình rồi chạy lại.")
        # nhánh count-only có w_d=0 nên không có density-MSE -> chỉnh w_count_mask là vô ích ở đó
        print("     " + ("count-only sập: w_count_mask KHÔNG liên quan (w_d=0). Hạ --lr / đổi khởi tạo."
                         if any("count" in b for b in bad) else
                         "mask sập: density-MSE bị nền chi phối -> tăng --w_count_mask, hoặc hạ --lr."))

    compare(per_img, gt_te, seeds, args, rows)


if __name__ == "__main__":
    main()
