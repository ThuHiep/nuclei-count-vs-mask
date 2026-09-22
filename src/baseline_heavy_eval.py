from __future__ import annotations
import argparse, csv, pickle
from collections import defaultdict
import numpy as np


def read_csv(path):
    with open(path) as f:
        r = csv.reader(f); header = next(r); return header, [row for row in r]


def load_gt(path):
    _, rows = read_csv(path)
    gt = {row[0]: float(row[1]) for row in rows}
    organ = {row[0]: (row[2] if len(row) > 2 else "_all_") for row in rows}
    return gt, organ


def load_preds(path, tiles_map=None):
    _, rows = read_csv(path)
    pred = {row[0]: float(row[1]) for row in rows}
    if tiles_map:
        _, trows = read_csv(tiles_map)
        agg = defaultdict(float)
        for tile, image in trows:
            agg[image] += pred.get(tile, 0.0)   # count ảnh = tổng count các tile
        return dict(agg)
    return pred


def stats(gt, pred, organ, label):
    keys = [k for k in gt if k in pred]
    miss = [k for k in gt if k not in pred]
    y = np.array([gt[k] for k in keys]); p = np.array([pred[k] for k in keys])
    ae = np.abs(p - y)
    mae = float(ae.mean()); rmse = float(np.sqrt(((p - y) ** 2).mean()))
    mape = float((ae / np.maximum(y, 1)).mean() * 100)
    bias = float((p - y).mean())   # signed: âm = under-predict (đếm thiếu), dương = over-predict
    # per-organ
    by = defaultdict(list)
    for k in keys:
        by[organ.get(k, "_all_")].append(abs(pred[k] - gt[k]))
    org_mae = {o: float(np.mean(v)) for o, v in by.items() if len(v) >= 5}
    worst = max(org_mae.items(), key=lambda kv: kv[1]) if org_mae else (None, None)
    return {"label": label, "n": len(keys), "miss": len(miss), "mae": mae, "rmse": rmse,
            "mape": mape, "bias": bias, "org_mae": org_mae, "worst": worst}


def dsunet_row(pkl, gt_from_pkl=True):
    """File dự đoán của DSU-Net: preds[{mu}], gts, organs -> sai số đếm trên chính tập kiểm tra của nó."""
    obj = pickle.load(open(pkl, "rb"))
    mu = np.array([float(pp["mu"]) for pp in obj["preds"]])
    y = np.array([float(np.asarray(g).reshape(-1)[0]) for g in obj["gts"]])
    organs = list(obj.get("organs", ["_all_"] * len(mu)))
    ae = np.abs(mu - y)
    by = defaultdict(list)
    for i, o in enumerate(organs):
        by[o].append(ae[i])
    org_mae = {o: float(np.mean(v)) for o, v in by.items() if len(v) >= 5}
    worst = max(org_mae.items(), key=lambda kv: kv[1]) if org_mae else (None, None)
    return {"label": "DSU-Net (3.62M)", "n": len(mu), "miss": 0,
            "mae": float(ae.mean()), "rmse": float(np.sqrt((ae ** 2).mean())),
            "mape": float((ae / np.maximum(y, 1)).mean() * 100), "bias": float((mu - y).mean()),
            "org_mae": org_mae, "worst": worst}




def pretty(rows):
    print("\n" + "=" * 84)
    print("Sai số đếm trên NuInsSeg (mạng nặng chạy sẵn, ngoài miền, so với DSU-Net trong miền)")
    print("=" * 84)
    h = (f"{'method':26} | {'N':>4} | {'MAE':>7} | {'RMSE':>7} | {'MAPE%':>6} | {'Bias':>8} | "
         f"{'worst-organ MAE':>22}")
    print(h); print("-" * len(h))
    for d in rows:
        wo = f"{d['worst'][0]}={d['worst'][1]:.2f}" if d["worst"][0] else "n/a"
        miss = f" (miss {d['miss']})" if d.get("miss") else ""
        print(f"{d['label']:26} | {d['n']:>4} | {d['mae']:7.2f} | {d['rmse']:7.2f} | "
              f"{d['mape']:6.1f} | {d.get('bias', 0.0):+8.2f} | {wo:>22}{miss}")
    print("-" * len(h))
    print("[ghi chu] MAE thap hon la dem tot hon. Cac mang nang chay o che do ngoai mien.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", default=None, help="preds.csv của heavy net (image,pred_count)")
    ap.add_argument("--tiles_map", default=None)
    ap.add_argument("--label", default="HeavyNet")
    ap.add_argument("--dsunet_pkl", default=None)
    args = ap.parse_args()

    gt, organ = load_gt(args.gt)
    rows = []
    if args.preds:
        pred = load_preds(args.preds, args.tiles_map)
        rows.append(stats(gt, pred, organ, args.label))
    if args.dsunet_pkl:
        rows.append(dsunet_row(args.dsunet_pkl))
    if not rows:
        print("Chưa có --preds hay --dsunet_pkl để chấm."); return
    pretty(rows)


if __name__ == "__main__":
    main()
