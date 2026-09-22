#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, glob, os, traceback
import numpy as np
from PIL import Image


def build_eval(model_type):
    """Trả eval_fn(img_np)->labels2d, bền với khác biệt API InstanSeg."""
    from instanseg import InstanSeg
    model = InstanSeg(model_type, verbosity=0)

    def _to_labels(out):
        import torch
        x = out[0] if isinstance(out, (list, tuple)) else out
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        while x.ndim > 2:          # (1,C,H,W)/(C,H,W) -> lấy channel 0 (nuclei)
            x = x[0]
        return x

    def _eval(img, pixel_size):
        try:
            out = model.eval_small_image(img, pixel_size=pixel_size)
        except TypeError:
            out = model.eval_small_image(img)   # version không nhận pixel_size
        return _to_labels(out)
    return _eval


def count_labels(lab):
    u = np.unique(lab)
    return int((u > 0).sum())


def load_done(out_csv):
    if not os.path.exists(out_csv):
        return set()
    with open(out_csv) as f:
        return {r["image"] for r in csv.DictReader(f) if r.get("pred_count", "") != ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model_type", default="brightfield_nuclei", help="H&E -> brightfield_nuclei")
    ap.add_argument("--pixel_size", type=float, default=0.5, help="µm/px (chỉnh nếu count vô lý)")
    ap.add_argument("--smoke", type=int, default=0, help=">0: chỉ N ảnh đầu (KHÔNG dùng cho bảng!)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images_dir, "*.png")))
    assert paths, f"không thấy .png trong {args.images_dir}"
    if args.smoke:
        print(f"SMOKE: chỉ {args.smoke}/{len(paths)} ảnh — KẾT QUẢ KHÔNG DÙNG cho bảng!")
        paths = paths[:args.smoke]

    done = load_done(args.out_csv)
    todo = [p for p in paths if os.path.splitext(os.path.basename(p))[0] not in done]
    print(f"{len(paths)} ảnh | đã xong {len(done)} | còn {len(todo)} | model={args.model_type} px={args.pixel_size}")
    if not todo:
        print("Tất cả đã xong."); return

    eval_fn = build_eval(args.model_type)

    err_log = args.out_csv + ".errors"
    new_file = not os.path.exists(args.out_csv)
    fout = open(args.out_csv, "a", newline="")
    w = csv.writer(fout)
    if new_file:
        w.writerow(["image", "pred_count"]); fout.flush()

    n_ok, n_err, errors = 0, 0, []
    for k, p in enumerate(todo):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            img = np.array(Image.open(p).convert("RGB"))
            count = count_labels(eval_fn(img, args.pixel_size))
            w.writerow([name, count]); fout.flush()
            n_ok += 1
        except Exception as e:
            n_err += 1; errors.append(name)
            with open(err_log, "a") as ef:
                ef.write(f"{name}\t{type(e).__name__}: {e}\n{traceback.format_exc()}\n")
            print(f"  LỖI {name}: {type(e).__name__}: {e} (ghi {err_log}, chạy tiếp)")
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(todo)}  (ok={n_ok} err={n_err})")
    fout.close()

    total = len(load_done(args.out_csv))
    print(f"\nXONG -> {args.out_csv} | thêm ok={n_ok} err={n_err} | tổng csv={total}/{len(paths)}")
    if total < len(paths):
        print(f"THIẾU {len(paths)-total} — xem {err_log}; chạy LẠI để resume.")


if __name__ == "__main__":
    main()
