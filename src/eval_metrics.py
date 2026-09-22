from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from losses import count_from_density


def eval_r2_mae(model, data, idx, dev):
    model.eval(); pr, yt = [], []
    for i in idx:
        x = torch.from_numpy(data[i]["img"].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
        pr.append(float(count_from_density(model(x)[0])[0])); yt.append(data[i]["gt"])
    pr, yt = np.array(pr), np.array(yt)
    ss_res = ((yt - pr) ** 2).sum(); ss_tot = ((yt - yt.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, np.abs(pr - yt).mean()


def winkler_score(lo: float, hi: float, y: float, alpha: float) -> float:
    """Interval score chuẩn (Gneiting & Raftery 2007) — dùng đúng ở paper 1."""
    w = hi - lo
    if y < lo:
        return w + (2.0 / alpha) * (lo - y)
    if y > hi:
        return w + (2.0 / alpha) * (y - hi)
    return w


def organ_conditional_stats(per_image_pooled: List[Tuple[str, int, bool, float]],
                            target: float, min_organ_imgs: int) -> Dict:
    """Gộp độ phủ theo cơ quan. Gộp qua các hạt giống để giảm phương sai của ước lượng, nhưng
    ngưỡng lọc tính theo số ảnh khác nhau chứ không theo số mẫu đã gộp, để không thổi phồng độ
    tin cậy ở những cơ quan hiếm."""
    by_organ_joint = defaultdict(list)
    by_organ_marg = defaultdict(list)
    by_organ_imgs = defaultdict(set)
    for organ, gidx, joint_cov, marg_cov in per_image_pooled:
        by_organ_joint[organ].append(1.0 if joint_cov else 0.0)
        by_organ_marg[organ].append(marg_cov)
        by_organ_imgs[organ].add(gidx)
    per_organ = {}
    for organ in by_organ_joint:
        n_imgs = len(by_organ_imgs[organ])
        if n_imgs < min_organ_imgs:
            continue  # quá ít ảnh distinct -> ước lượng coverage không tin cậy, bỏ
        per_organ[organ] = {
            "n_distinct_imgs": n_imgs,
            "n_pooled": len(by_organ_joint[organ]),
            "joint_coverage": float(np.mean(by_organ_joint[organ])),
            "marginal_coverage": float(np.mean(by_organ_marg[organ])),
        }
    if not per_organ:
        return {"per_organ": {}, "worst_organ_coverage": None,
                "best_organ_coverage": None, "organ_coverage_gap": None,
                "n_organs_undercovered": 0, "n_organs_eval": 0}
    covs = {o: per_organ[o]["joint_coverage"] for o in per_organ}
    worst_o = min(covs, key=covs.get)
    best_o = max(covs, key=covs.get)
    under = sum(1 for c in covs.values() if c < target - 0.05)  # <target-5% coi là under
    return {
        "per_organ": per_organ,
        "worst_organ": worst_o,
        "worst_organ_coverage": float(covs[worst_o]),
        "best_organ_coverage": float(covs[best_o]),
        "organ_coverage_gap": float(covs[best_o] - covs[worst_o]),
        "n_organs_undercovered": int(under),
        "n_organs_eval": len(per_organ),
    }
