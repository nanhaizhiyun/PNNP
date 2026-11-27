# utf-8
"""用 NAFNet(RawGT) 批量处理 DNG 目录，输出 JPG（完整显示管线）。

显示链路与 scripts/inference_pnc460.py 一致：
分位数增益+HDR保护 -> 高光软削顶 -> 网络 -> 亮像素旁路
-> 灰世界白平衡 -> 去马赛克 -> 自适应gamma -> 显示域色度抑制 -> JPG
"""
import os, sys, glob, argparse, importlib.util
import numpy as np
import cv2
import rawpy
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from archs.NAFNet import NAFNet
from utils import raw2bayer

spec = importlib.util.spec_from_file_location('inf', os.path.join(REPO, 'scripts/inference_pnc460.py'))
inf = importlib.util.module_from_spec(spec); spec.loader.exec_module(inf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', '-i', required=True)
    ap.add_argument('--output', '-o', required=True)
    ap.add_argument('--ckpt', default='/root/chenbf/pnnp_out/checkpoints/PNC460/'
                                      'PNC460_on_RawGT_NAFNet_best_model.pth')
    ap.add_argument('--ratio', type=float, default=0.0, help='>0 时固定增益')
    ap.add_argument('--ratio-min', type=float, default=20.0)
    ap.add_argument('--ratio-max', type=float, default=80.0)
    ap.add_argument('--quality', type=int, default=95)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    net = NAFNet({'in_nc': 4, 'out_nc': 4, 'nf': 32,
                  'enc_blk_nums': [2, 2, 4, 8], 'middle_blk_num': 12,
                  'dec_blk_nums': [2, 2, 2, 2]})
    net.load_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=True), strict=True)
    net = net.cuda().eval()

    files = sorted(glob.glob(os.path.join(args.input, '*.dng')))
    if not files:
        print('未找到 DNG'); return

    for src in files:
        r = rawpy.imread(src)
        raw = r.raw_image_visible.astype(np.float32)
        bl = float(np.mean(r.black_level_per_channel))
        wp = float(r.white_level)

        x = np.clip((raw2bayer(raw, norm=False) - bl) / (wp - bl), 0, 1)
        if args.ratio > 0:
            ratio, info = args.ratio, f'固定 {args.ratio:.0f}'
        else:
            ratio, info = inf.compute_gain(x, ratio_min=args.ratio_min, ratio_max=args.ratio_max)
        xin = x * ratio
        with torch.no_grad():
            pred = net(torch.from_numpy(inf.soft_clip_highlights(xin))[None].cuda()).clamp(0, 1)[0].cpu().numpy()
        pred = np.where(xin > 0.85, np.clip(xin, 0, 1), pred)

        # 白平衡：亮区主导灰世界
        lum = pred.mean(axis=0)
        sel = lum >= np.percentile(lum, 90)
        rm = float(pred[0][sel].mean()); gm = float((pred[1][sel].mean() + pred[3][sel].mean()) / 2)
        bm = float(pred[2][sel].mean())
        wb = np.array([gm / max(rm, 1e-6), 1.0, gm / max(bm, 1e-6)])

        # 显示链路：去马赛克 -> p99.5 归一化 -> 自适应gamma -> 色度抑制
        # （模型输出动态范围很低（mean≈0.02），直接 gamma 会偏暗；
        #   先按显示图 p99.5 拉伸到满量程再算 gamma，亮度才正常）
        rgb = inf.demosaic(pred, wb)
        rgb = np.clip(rgb, 0, 1)
        p995 = np.percentile(rgb, 99.5)
        rgb = np.clip(rgb / max(p995, 1e-4), 0, 1)
        g = np.log(0.42) / np.log(max(float(rgb.mean()), 1e-4))
        rgb = rgb ** g
        s_chroma = inf.auto_chroma_strength(float(raw.std()))
        if s_chroma < 1.0:
            rgb = inf.chroma_denoise_display(rgb, s_chroma)

        out = os.path.join(args.output, os.path.basename(src).replace('.dng', '.jpg'))
        cv2.imwrite(out, np.uint8(np.clip(rgb, 0, 1)[..., ::-1] * 255),
                    [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        print(f'{os.path.basename(src)}: ratio={ratio:.0f} ({info}) p995={p995:.3f} gamma={g:.2f} '
              f'chroma={s_chroma:.2f} -> {out}')
        r.close()


if __name__ == '__main__':
    main()
