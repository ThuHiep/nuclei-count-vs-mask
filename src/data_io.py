import glob, os
from pathlib import Path

import numpy as np
from PIL import Image

REPO = os.environ.get("REPO", ".")

IMG_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")

NUINSSEG_CANDS = [f"{REPO}/data/nuinsseg", f"{REPO}/data/nuinsseg/NuInsSeg",
                  "data/NuInsSeg"]

IMG_SIZE = 256


def _find_mask_dir(organ_dir):
    for name in os.listdir(organ_dir):
        full = os.path.join(organ_dir, name); low = name.lower()
        if os.path.isdir(full) and "label" in low and "mask" in low and "modif" not in low:
            return full
    for name in os.listdir(organ_dir):
        full = os.path.join(organ_dir, name)
        if os.path.isdir(full) and "label" in name.lower():
            return full
    return None


def _load_mask(path):
    try:
        import tifffile
        if path.lower().endswith((".tif", ".tiff")):
            return np.asarray(tifffile.imread(path))
    except Exception:
        pass
    return np.asarray(Image.open(path))


def build_index(root):
    tissue_dirs = glob.glob(os.path.join(root, "**", "tissue images"), recursive=True)
    samples = []
    for tdir in tissue_dirs:
        organ_dir = os.path.dirname(tdir); organ = os.path.basename(organ_dir)
        mdir = _find_mask_dir(organ_dir)
        if mdir is None:
            continue
        masks = {os.path.splitext(f)[0]: os.path.join(mdir, f) for f in os.listdir(mdir)}
        for f in sorted(os.listdir(tdir)):
            if not f.lower().endswith(IMG_EXT):
                continue
            stem = os.path.splitext(f)[0]
            if stem in masks:
                samples.append({"organ": organ, "image": os.path.join(tdir, f),
                                "mask": masks[stem]})
    return samples


def find_root():
    root = next((c for c in NUINSSEG_CANDS if os.path.isdir(c)), None)
    if root is None:
        td = glob.glob(f"{REPO}/data/**/tissue images", recursive=True)
        root = os.path.dirname(os.path.dirname(td[0])) if td else None
    assert root, ("Khong tim thay NuInsSeg. Dat duong dan qua bien moi truong REPO "
                  "--unzip -p data/nuinsseg/")
    return root


def assign_kfold(organs, k, seed):
    """Gán fold [0..k) cho mỗi ảnh, PHÂN TẦNG theo organ (mỗi organ trải đều k fold).
    -> mỗi fold có phân bố organ ~giống nhau, tốt cho conditional coverage. Trả np.int array (N,)."""
    from collections import defaultdict
    rng = np.random.RandomState(seed)
    fold_of = np.zeros(len(organs), dtype=int)
    by_org = defaultdict(list)
    for i, o in enumerate(organs):
        by_org[o].append(i)
    for t, (o, idxs) in enumerate(sorted(by_org.items())):
        idxs = list(idxs); rng.shuffle(idxs)
        for j, i in enumerate(idxs):
            fold_of[i] = (j + t) % k   # xoay start theo organ -> phần dư trải đều, fold ~cân
    return fold_of


def _fold_base(root, fold):
    f = f"fold{fold}"
    for c in (root / f / f"Fold {fold}", root / f"Fold {fold}"):
        if (c / "images" / f / "images.npy").exists():
            return c
    raise FileNotFoundError(f"Không thấy Fold {fold} dưới {root}")


def _pannuke_counts(base, fold):
    d = base / "images" / f"fold{fold}"
    if (d / "counts.npy").exists():
        return np.load(d / "counts.npy")
    cache = Path.cwd() / f"pannuke_counts_fold{fold}.npy"
    if cache.exists():
        return np.load(cache)
    mpath = base / "masks" / f"fold{fold}" / "masks.npy"
    print(f"[pannuke] tính counts từ {mpath} (một lần) ...")
    masks = np.load(mpath, mmap_mode="r"); n = masks.shape[0]
    counts = np.zeros((n, 5), np.int32)
    for i in range(n):
        m = np.asarray(masks[i, :, :, :5], np.int32)
        for k in range(5):
            counts[i, k] = int(np.unique(m[:, :, k]).size - 1)
    try:
        np.save(cache, counts)
    except Exception:
        pass
    return counts


def load_pannuke_data(root, folds, exclude, max_imgs, seed=0):
    """Trả list dict {img uint8 256, density zeros, gt, organ} — đúng format train()/eval."""
    root = Path(root); out = []
    for fold in folds:
        base = _fold_base(root, fold)
        d = base / "images" / f"fold{fold}"
        imgs = np.load(d / "images.npy", mmap_mode="r")
        tp = d / "types.npy"
        types = np.load(tp, allow_pickle=True) if tp.exists() else np.array(["na"] * len(imgs))
        gt = _pannuke_counts(base, fold).sum(1).astype(np.float32)
        for i in range(len(imgs)):
            t = str(types[i])
            if exclude and exclude.lower() in t.lower():
                continue
            im = np.asarray(imgs[i])
            if im.max() <= 1.5:
                im = (im * 255)
            out.append({"img": im.astype(np.uint8),
                        "density": np.zeros((IMG_SIZE, IMG_SIZE), np.float32),
                        "gt": float(gt[i]), "organ": t})
    if max_imgs and len(out) > max_imgs:
        r = np.random.default_rng(seed).choice(len(out), max_imgs, replace=False)
        out = [out[i] for i in r]
    return out
