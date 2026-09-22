from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    import torch
    from dsunet import DensitySigmaUNet
    assert torch.cuda.is_available(), "cần GPU"
    dev = "cuda"; gpu = torch.cuda.get_device_name(0)
    m = DensitySigmaUNet(ch=args.ch).to(dev).eval()
    nparam = sum(p.numel() for p in m.parameters())

    def bench(bs, iters, warm=20):
        x = torch.randn(bs, 3, args.size, args.size, device=dev)
        for _ in range(warm):
            with torch.no_grad():
                m(x)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.time()
        for _ in range(iters):
            with torch.no_grad():
                m(x)
        torch.cuda.synchronize()
        dt = (time.time() - t) / iters
        return dt, torch.cuda.max_memory_allocated() / 1e6

    print(f"GPU={gpu} | DensitySigmaUNet ch={args.ch} params={nparam/1e6:.3f}M | size={args.size}")
    dt1, vram1 = bench(1, args.iters)
    print(f"[bs=1]  latency = {dt1*1000:.2f} ms/img | throughput = {1/dt1:.1f} img/s | peak VRAM = {vram1:.1f} MB")
    dtb, vramb = bench(args.batch, max(args.iters // 4, 20))
    print(f"[bs={args.batch}] {dtb*1000:.1f} ms/batch | throughput = {args.batch/dtb:.0f} img/s | peak VRAM = {vramb:.1f} MB")
    print("[ghi chú] chỉ đo lượt truyền xuôi, chưa gồm tiền/hậu xử lý. "
          "(bảng A) vì không dựng lại env CellViT/LKCell ở đây — trung thực.")


if __name__ == "__main__":
    main()
