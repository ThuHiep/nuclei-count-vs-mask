from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=32, help="số kênh của bộ giải mã (32 = cấu hình dùng trong bài)")
    ap.add_argument("--size", type=int, default=256, help="cạnh ảnh vuông cho MACs")
    args = ap.parse_args()

    import torch
    from dsunet import DensitySigmaUNet  # model thật (single source of truth)

    m = DensitySigmaUNet(ch=args.ch).eval()
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"DensitySigmaUNet(ch={args.ch}): params total = {tot:,} ({tot/1e6:.3f} M), trainable = {tr:,}")

    try:
        from thop import profile
        x = torch.randn(1, 3, args.size, args.size)
        macs, _ = profile(m, inputs=(x,), verbose=False)
        print(f"MACs @{args.size} = {macs/1e9:.3f} G   (= {macs/1e9:.2f} GMACs; báo cột GFLOPs của paper = MACs)")
        print(f"(nếu paper thật sự dùng FLOPs=2xMACs thì = {2*macs/1e9:.3f} G — kiểm quy ước trước khi in)")
    except ModuleNotFoundError:
        print("thop chưa cài -> `pip install thop` để lấy GMACs. (params ở trên vẫn đúng)")

    print(f"\n  DSU-Net | params {tot/1e6:.3f} M | GMACs@256 ở dòng trên")
    print("  Số của các mạng nặng lấy từ bài gốc của chúng, không đo lại:")
    print("  CellViT-SAM-H 699.74M / 214.33 | LKCell-L 163.84M / 47.86 | NuLite-T 17.12M / 26.16")


if __name__ == "__main__":
    main()
