import torch
from losses import (count_from_density, density_kd_loss, count_loss,
                       gaussian_nll, poisson_nll, r2_loss)

torch.manual_seed(0)
N_PASS = 0


def ok(name, cond):
    global N_PASS
    assert cond, f"FAIL: {name}"
    N_PASS += 1
    print(f"  ok  {name}")


def test_count_from_density():
    d = torch.rand(4, 1, 16, 16)
    mu = count_from_density(d)
    ok("count_from_density shape", mu.shape == (4,))
    ok("count == manual sum", torch.allclose(mu, d.sum(dim=(1, 2, 3))))
    ok("count_from_density (B,H,W)", count_from_density(torch.rand(3, 8, 8)).shape == (3,))


def test_density_kd_grad():
    dS = torch.rand(2, 1, 16, 16, requires_grad=True)
    dT = torch.rand(2, 1, 16, 16)
    L = density_kd_loss(dS, dT)
    L.backward()
    ok("density_kd differentiable", dS.grad is not None and dS.grad.abs().sum() > 0)


def test_gaussian_nll_formula():
    mu = torch.tensor([10.0]); ls = torch.tensor([0.0]); gt = torch.tensor([12.0])
    # beta=0 -> NLL = 0.5*[(2)^2/1 + 0] = 2.0
    val = gaussian_nll(mu, ls, gt, beta=0.0)
    ok("gaussian_nll beta=0 formula", torch.allclose(val, torch.tensor(2.0), atol=1e-5))
    # sigma=e^1: var=e^2; NLL=0.5*[(4)/e^2 + 2]
    val2 = gaussian_nll(torch.tensor([10.0]), torch.tensor([1.0]), torch.tensor([12.0]), beta=0.0)
    exp2 = 0.5 * (4.0 / (2.718281828 ** 2) + 2.0)
    ok("gaussian_nll sigma>1 formula", abs(float(val2) - exp2) < 1e-4)


def test_beta_nll_grad_mu_nonzero():
    # beta-NLL: gradient của mu không bị triệt tiêu khi sigma nhỏ
    mu = torch.tensor([10.0], requires_grad=True)
    ls = torch.tensor([-2.0])  # sigma nhỏ
    gt = torch.tensor([12.0])
    gaussian_nll(mu, ls, gt, beta=0.5).backward()
    ok("beta-NLL grad mu nonzero", mu.grad is not None and mu.grad.abs().item() > 1e-6)


def test_nll_grad_logsigma():
    mu = torch.tensor([10.0]); ls = torch.tensor([0.5], requires_grad=True); gt = torch.tensor([12.0])
    gaussian_nll(mu, ls, gt, beta=0.5).backward()
    ok("NLL grad log_sigma nonzero", ls.grad is not None and ls.grad.abs().item() > 1e-6)


def test_sigma_calibrates():
    # Cốt lõi: tối ưu NLL -> sigma hội tụ về ~|gt-mu| (calibrated). Cố định mu, học log_sigma.
    mu = torch.full((256,), 10.0)
    gt = mu + torch.randn(256) * 3.0            # lỗi thật std ~3
    ls = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([ls], lr=0.05)
    for _ in range(2000):
        opt.zero_grad()
        loss = gaussian_nll(mu, ls.expand(256), gt, beta=0.0)
        loss.backward(); opt.step()
    sigma = float(torch.exp(ls))
    # NLL tối ưu tại sigma^2 = E[(gt-mu)^2] -> sigma ~ std thật (~3)
    ok(f"sigma calibrates to true error (sigma={sigma:.2f}, target~3)", 2.0 < sigma < 4.0)


def test_heteroscedastic_learns_perinput():
    # 2 nhóm: nhóm A lỗi nhỏ, nhóm B lỗi lớn -> log_sigma head học sigma_B > sigma_A
    muA = torch.full((128,), 5.0); gtA = muA + torch.randn(128) * 1.0
    muB = torch.full((128,), 20.0); gtB = muB + torch.randn(128) * 6.0
    lsA = torch.zeros(1, requires_grad=True)
    lsB = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([lsA, lsB], lr=0.05)
    for _ in range(2000):
        opt.zero_grad()
        loss = (gaussian_nll(muA, lsA.expand(128), gtA, beta=0.0)
                + gaussian_nll(muB, lsB.expand(128), gtB, beta=0.0))
        loss.backward(); opt.step()
    sA, sB = float(torch.exp(lsA)), float(torch.exp(lsB))
    ok(f"heteroscedastic: sigma_hard>sigma_easy ({sB:.2f}>{sA:.2f})", sB > sA + 1.0)


def test_r2_loss_end2end_grad():
    dS = torch.rand(4, 1, 16, 16, requires_grad=True)
    dT = torch.rand(4, 1, 16, 16)
    ls = torch.randn(4, requires_grad=True)
    gt = torch.tensor([12.0, 5.0, 30.0, 0.0])
    out = r2_loss(dS, dT, ls, gt, w_density=1.0, w_count=1.0, w_nll=1.0, beta=0.5)
    out["loss"].backward()
    ok("r2_loss grad -> density head", dS.grad is not None and dS.grad.abs().sum() > 0)
    ok("r2_loss grad -> log_sigma head", ls.grad is not None and ls.grad.abs().sum() > 0)
    ok("r2_loss keys", {"loss", "L_density", "L_count", "L_nll", "mu", "sigma"} <= set(out))
    ok("r2_loss mu == sum density", torch.allclose(out["mu"], dS.detach().sum(dim=(1, 2, 3))))


def test_edge_cases():
    # gt=0, density~0
    dS = torch.zeros(1, 1, 8, 8, requires_grad=True)
    dT = torch.zeros(1, 1, 8, 8)
    ls = torch.zeros(1, requires_grad=True)
    gt = torch.zeros(1)
    out = r2_loss(dS, dT, ls, gt)
    ok("edge gt=0 finite", torch.isfinite(out["loss"]))
    out["loss"].backward()
    ok("edge gt=0 grad finite", torch.isfinite(ls.grad).all())
    # poisson nll finite khi mu->0
    ok("poisson_nll clamps mu>0", torch.isfinite(poisson_nll(torch.zeros(3), torch.tensor([1., 2., 3.]))))


if __name__ == "__main__":
    for fn in [test_count_from_density, test_density_kd_grad, test_gaussian_nll_formula,
               test_beta_nll_grad_mu_nonzero, test_nll_grad_logsigma, test_sigma_calibrates,
               test_heteroscedastic_learns_perinput, test_r2_loss_end2end_grad, test_edge_cases]:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\n{N_PASS}/{N_PASS} PASS")
