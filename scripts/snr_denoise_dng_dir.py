# utf-8
"""用 SNR 版模型批量处理 DNG 目录，输出 DNG（保留源文件的位深/黑电平/色彩矩阵）。

用法：
  python3 scripts/snr_denoise_dng_dir.py --input /root/chenbf/DngInput \
      --output /root/chenbf/pnnp_out/snr_on_dnginput [--ratio 60]

比例（ratio）默认按训练分布 20-80 自动选取；近噪声底场景会落在 80。
"""
import os, sys, glob, argparse, importlib.util
import numpy as np
import rawpy
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from archs.Unet import UNetSeeInDark
from utils import load_weights, raw2bayer

spec = importlib.util.spec_from_file_location('inf', os.path.join(REPO, 'scripts/inference_pnc460.py'))
inf = importlib.util.module_from_spec(spec); spec.loader.exec_module(inf)
spec2 = importlib.util.spec_from_file_location('dng', os.path.join(REPO, 'scripts/snr_denoise_sid_val_to_dng.py'))
dngmod = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(dngmod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', '-i', required=True)
    ap.add_argument('--output', '-o', required=True)
    ap.add_argument('--ckpt', default='/root/chenbf/pnnp_out/checkpoints/PNC460/'
                                      'PNC460_on_CombGT_SNR_Unet_best_model.pth')
    ap.add_argument('--ratio', type=float, default=0.0, help='>0 时固定增益')
    ap.add_argument('--ratio-min', type=float, default=20.0)
    ap.add_argument('--ratio-max', type=float, default=80.0)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    net = load_weights(
        UNetSeeInDark({'in_nc': 4, 'out_nc': 4, 'nf': 32, 'nframes': 1, 'use_dpsv': False,
                       'res': False, 'cascade': False, 'add': False, 'lock_wb': False}),
        torch.load(args.ckpt, map_location='cpu', weights_only=True), by_name=True).cuda().eval()

    files = sorted(glob.glob(os.path.join(args.input, '*.dng')))
    if not files:
        print('未找到 DNG'); return

    for src in files:
        r = rawpy.imread(src)
        raw = r.raw_image_visible.astype(np.float32)
        bl = float(np.mean(r.black_level_per_channel))
        wp = float(r.white_level)
        try:
            cam2xyz = np.array(r.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
            color_mat = np.linalg.inv(cam2xyz)
        except Exception:
            color_mat = None

        x = np.clip((raw2bayer(raw, norm=False) - bl) / (wp - bl), 0, 1)
        if args.ratio > 0:
            ratio, info = args.ratio, f'固定 {args.ratio:.0f}'
        else:
            ratio, info = inf.compute_gain(x, ratio_min=args.ratio_min, ratio_max=args.ratio_max)
        xin = x * ratio
        with torch.no_grad():
            pred = net(torch.from_numpy(inf.soft_clip_highlights(xin))[None].cuda()).clamp(0, 1)[0].cpu().numpy()
        pred = np.where(xin > 0.85, np.clip(xin, 0, 1), pred)

        # 色度噪声抑制：低光下 R/B 通道几乎全是噪声，直接写 raw 会出现
        # 绿/品红色斑（显示管线里的 CNR 在这里必须提前做）
        s_chroma = inf.auto_chroma_strength(float(raw.std()))
        if s_chroma < 1.0:
            R, G, B = pred[0], (pred[1] + pred[3]) / 2, pred[2]
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            Cb = -0.168736 * R - 0.331264 * G + 0.5 * B
            Cr = 0.5 * R - 0.418688 * G - 0.081312 * B
            Cb, Cr = Cb * s_chroma, Cr * s_chroma
            R2 = Y + 1.402 * Cr
            G2 = Y - 0.344136 * Cb - 0.714136 * Cr
            B2 = Y + 1.772 * Cb
            pred = np.stack([R2, G2, B2, G2])

        # 灰世界白平衡（模型输出平衡 ≠ 传感器原始响应）
        rm, gm, bm = float(pred[0].mean()), float((pred[1].mean() + pred[3].mean()) / 2), float(pred[2].mean())
        wb = np.array([gm / max(rm, 1e-6), 1.0, gm / max(bm, 1e-6), 1.0])

        # 放大域写回 raw，重拼 Bayer
        pk = np.clip(pred, 0, 1) * (wp - bl) + bl
        H, W = pk.shape[1] * 2, pk.shape[2] * 2
        bayer = np.empty((H, W), np.float32)
        bayer[0::2, 0::2] = pk[0]; bayer[0::2, 1::2] = pk[1]
        bayer[1::2, 1::2] = pk[2]; bayer[1::2, 0::2] = pk[3]

        out = os.path.join(args.output, os.path.basename(src))
        dngmod.save_dng(out, np.clip(bayer, 0, wp).astype(np.uint16),
                        black=bl, white=wp, make='Hytera', model='PNC460',
                        wb=wb, color_matrix=color_mat)
        chk = rawpy.imread(out)
        print(f'{os.path.basename(src)}: ratio={ratio:.0f} ({info}) bl={bl:.0f} wp={wp:.0f} '
              f'std={raw.std():.1f} -> {out} (mean={chk.raw_image_visible.mean():.0f})')
        chk.close(); r.close()


if __name__ == '__main__':
    main()
