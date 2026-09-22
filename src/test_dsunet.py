import os, pickle, sys, tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dsunet import DensitySigmaUNet, train, predict_r2  # noqa: E402

N_PASS = 0
def ok(name, cond):
    global N_PASS
    assert cond, f"FAIL: {name}"
    N_PASS += 1
    print(f"  ok  {name}")


def make_synth(n=48, size=64, organs=("A", "B", "C")):
    """Synthetic: density map = vài blob, gt ~ tổng blob; organ B khó (nhiễu lớn)."""
    rng = np.random.RandomState(0)
    data = []
    for i in range(n):
        organ = organs[i % len(organs)]
        k = rng.randint(3, 20)                       # số nhân
        dens = np.zeros((size, size), np.float32)
        for _ in range(k):
            y, x = rng.randint(4, size - 4, size=2)
            dens[y - 2:y + 2, x - 2:x + 2] += 1.0 / 16.0   # mỗi nhân khối lượng 1
        img = (dens[..., None].repeat(3, 2) * 200 + rng.rand(size, size, 3) * 40).clip(0, 255)
        gt = float(k)
        data.append({"img": img.astype(np.uint8), "density": dens, "gt": gt, "organ": organ})
    return data


def test_forward():
    m = DensitySigmaUNet(ch=16, backbone="tinyunet")   # tinyunet: khong can timm/tai pretrained -> test chay offline
    n_params = sum(p.numel() for p in m.parameters())
    x = torch.rand(2, 3, 64, 64)
    dens, ls = m(x)
    ok(f"forward density shape ({tuple(dens.shape)})", dens.shape == (2, 1, 64, 64))
    ok("density >= 0 (softplus)", bool((dens >= 0).all()))
    ok(f"log_sigma shape ({tuple(ls.shape)})", ls.shape == (2,))
    ok(f"params finite ~{n_params/1e6:.3f}M", 0 < n_params < 5e6)


def test_train_predict():
    data = make_synth()
    device = "cpu"
    # so mat mat dau va cuoi de xac nhan mo hinh co hoc; khong doi hoi tu o day
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = train(data, device, epochs=20, ch=16, lr=1e-3, train_idx=list(range(len(data))),
                      w_density=1.0, w_count=0.01, w_nll=0.01, beta=0.5, bs=8)
    log = buf.getvalue(); print(log, end="")
    losses = [float(l.split("loss=")[1].split()[0]) for l in log.splitlines() if "loss=" in l]
    ok(f"train loss giảm ({losses[0]:.1f} -> {losses[-1]:.1f})", losses[-1] < losses[0])
    out = predict_r2(model, data, device)
    ok("predict schema keys", set(out) == {"preds", "gts", "organs"})
    ok("pred has mu/sigma", all("mu" in p and "sigma" in p for p in out["preds"]))
    mu = np.array([p["mu"] for p in out["preds"]])
    sg = np.array([p["sigma"] for p in out["preds"]])
    gt = np.array([g[0] for g in out["gts"]])
    ok("mu finite", np.isfinite(mu).all())
    ok("sigma > 0", (sg > 0).all())
    ok(f"sigma heteroscedastic (std={sg.std():.2f}>0)", sg.std() > 1e-3)
    ok(f"count MAE finite & bounded (MAE={np.abs(mu-gt).mean():.1f})", np.abs(mu - gt).mean() < 50)
    return out


def test_eval(out):
    from prediction_intervals import run as eval_run
    tmp = tempfile.mkdtemp()
    r2_path = os.path.join(tmp, "r2.pkl")
    pickle.dump(out, open(r2_path, "wb"))
    # fake KD pkl (scores schema) cùng GT/organ để so 2 model
    kd = {"preds": [], "gts": out["gts"], "organs": out["organs"]}
    rng = np.random.RandomState(1)
    for g in out["gts"]:
        n = max(1, int(round(g[0] + rng.randn() * 3)))
        kd["preds"].append({"scores": np.clip(rng.rand(n) * 0.4 + 0.6, 0, 1).astype(np.float32),
                            "probs": np.ones((n, 1), np.float32), "K": 1})
    kd_path = os.path.join(tmp, "kd.pkl")
    pickle.dump(kd, open(kd_path, "wb"))
    res = eval_run([kd_path, r2_path], ["KD", "R2"], alpha=0.1, seeds=10,
                   cal_ratio=0.5, min_organ_imgs=2)
    for lab in ("KD", "R2"):
        cov = res["models"][lab]["coverage"]["mean"]
        ok(f"{lab} marginal coverage ~target (0.9) got {cov:.3f}", 0.75 <= cov <= 1.0)
        ok(f"{lab} winkler finite", np.isfinite(res["models"][lab]["winkler"]["mean"]))
    from prediction_intervals import pretty
    pretty(res)          # in bảng so sánh, chỉ kiểm là chạy không lỗi
    ok("in bảng khoảng dự báo không lỗi", True)


if __name__ == "__main__":
    print("[test_forward]"); test_forward()
    print("[test_train_predict]"); out = test_train_predict()
    print("[test_eval]"); test_eval(out)
    print(f"\n{N_PASS}/{N_PASS} PASS")
