# utf-8
"""用 SNR 版模型处理 SID Sony 验证集，输出保存为 DNG。

输入：/root/chenbf/SID/Sony/short/<scene>_00_0.1s.ARW（短曝光噪声帧）
输出：<out>/<scene>_denoised.dng（16-bit RGGB DNG，可被 rawpy/各类 raw 软件打开）

前处理/后处理与 scripts/inference_pnc460.py 一致；ratio 限制在训练分布 20-80。
"""
import os, sys, argparse, glob
import numpy as np
import rawpy
import torch
import tifffile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from archs.Unet import UNetSeeInDark
from utils import load_weights, raw2bayer
import importlib.util
spec = importlib.util.spec_from_file_location('inf', os.path.join(REPO, 'scripts/inference_pnc460.py'))
inf = importlib.util.module_from_spec(spec); spec.loader.exec_module(inf)

# Sony A7S2
BL, WP = 512, 16383
SCENES = ['10003', '10006', '10011', '10016', '10022']


def _cm18(m):
    """3x3 矩阵 -> TIFF RATIONAL 序列（18 个整数，分母 10000）。"""
    if m is None:
        m = np.eye(3)
    vals = []
    for v in np.asarray(m, dtype=np.float64).flatten():
        vals += [int(round(v * 10000)), 10000]
    return tuple(vals)


def save_dng(path, raw_u16, black=BL, white=WP, make='SONY', model='ILCE-7S', wb=None, color_matrix=None):
    """写 16-bit RGGB DNG（TIFF 容器 + DNG/CFA 标签）。"""
    extratags = [
        (271, 2, 0, make, True),                        # Make (ASCII)
        (272, 2, 0, model, True),                       # Model
        (33421, 3, 2, (2, 2), True),                    # CFARepeatPatternDim (SHORT)
        (33422, 1, 4, (0, 1, 1, 2), True),              # CFAPattern RGGB (BYTE)
        (50706, 1, 4, (1, 4, 0, 0), True),              # DNGVersion 1.4.0.0
        (50707, 1, 4, (1, 1, 0, 0), True),              # DNGBackwardVersion
        (50708, 2, 0, f'{make} {model}', True),         # UniqueCameraModel
        (50714, 5, 1, (int(black), 1), True),           # BlackLevel (RATIONAL)
        (50717, 5, 1, (int(white), 1), True),           # WhiteLevel
        # ColorMatrix1：XYZ -> 相机原生，9 个 RATIONAL（18 个整数）
        (50721, 10, 9, _cm18(color_matrix), True),   # SRATIONAL（含负值）
        (50778, 3, 1, 21, True),                        # CalibrationIlluminant1 = D65
    ]
    if wb is not None:
        # AsShotNeutral：告诉 raw 软件 R:G:B 的中性比值（=1/白平衡增益），
        # 缺少该标签时渲染会按无白平衡处理 -> 画面偏绿
        wn = (int(round(1.0 / wb[0] * 10000)), 10000,
              1, 1,
              int(round(1.0 / wb[2] * 10000)), 10000)
        extratags.append((50728, 5, 3, wn, True))

    tifffile.imwrite(path, raw_u16, photometric='CFA', compression=None,
                     extratags=extratags, metadata=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/root/chenbf/pnnp_out/checkpoints/PNC460/'
                                      'PNC460_on_CombGT_SNR_Unet_best_model.pth')
    ap.add_argument('--out', default='/root/chenbf/pnnp_out/snr_on_sid_val')
    ap.add_argument('--ratio-min', type=float, default=20.0)
    ap.add_argument('--ratio-max', type=float, default=80.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    net = load_weights(
        UNetSeeInDark({'in_nc': 4, 'out_nc': 4, 'nf': 32, 'nframes': 1, 'use_dpsv': False,
                       'res': False, 'cascade': False, 'add': False, 'lock_wb': False}),
        torch.load(args.ckpt, map_location='cpu', weights_only=True), by_name=True).cuda().eval()

    for scene in SCENES:
        cands = sorted(glob.glob(f'/root/chenbf/SID/Sony/short/{scene}_00_*.ARW'))
        if not cands:
            print(f'[跳过] {scene}: 无短曝光文件'); continue
        src = [c for c in cands if '0.1s' in c] or cands
        src = src[0]
        r = rawpy.imread(src)
        raw = r.raw_image_visible.astype(np.float32)
        wb_cam = np.array(r.camera_whitebalance, dtype=np.float64)
        wb_cam = wb_cam / wb_cam[1]          # (R, 1, B, 1)
        # DNG ColorMatrix1 = (camera->XYZ) 的逆，取自源 ARW 的真实色彩矩阵
        try:
            cam2xyz = np.array(r.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
            color_mat = np.linalg.inv(cam2xyz)
        except Exception:
            color_mat = None
        x = np.clip((raw2bayer(raw, norm=False) - BL) / (WP - BL), 0, 1)
        ratio, info = inf.compute_gain(x, ratio_min=args.ratio_min, ratio_max=args.ratio_max)
        xin = x * ratio
        xn = inf.soft_clip_highlights(xin)
        with torch.no_grad():
            pred = net(torch.from_numpy(xn)[None].cuda()).clamp(0, 1)[0].cpu().numpy()
        pred = np.where(xin > 0.85, np.clip(xin, 0, 1), pred)

        # 保存为"放大域"（模型输出本身，与 SID 官方做法一致）：
        # raw = pred*(wp-bl)+bl。若需要原始曝光尺度，除以 ratio 即可
        raw_out = np.clip(pred, 0, 1) * (WP - BL) + BL
        # 白平衡：模型输出的通道平衡已不等于源 ARW 的原始响应
        # （网络学到的是"干净图"的平衡），因此按输出自身做灰世界，
        # 写进 AsShotNeutral，保证任何 raw 软件打开都不偏色
        rm = float(pred[0].mean()); gm = float((pred[1].mean() + pred[3].mean()) / 2)
        bm = float(pred[2].mean())
        wb_cam = np.array([gm / max(rm, 1e-6), 1.0, gm / max(bm, 1e-6), 1.0])
        packed = raw_out                                 # (4,h,w)
        H, W = packed.shape[1] * 2, packed.shape[2] * 2
        bayer = np.empty((H, W), dtype=np.float32)
        bayer[0::2, 0::2] = packed[0]; bayer[0::2, 1::2] = packed[1]
        bayer[1::2, 1::2] = packed[2]; bayer[1::2, 0::2] = packed[3]
        out_path = os.path.join(args.out, f'{scene}_denoised.dng')
        save_dng(out_path, np.clip(bayer, 0, WP).astype(np.uint16), wb=wb_cam, color_matrix=color_mat)

        # 立即回读验证
        chk = rawpy.imread(out_path)
        print(f'{scene}: 输入 {os.path.basename(src)} ratio={ratio:.0f} ({info}) '
              f'-> {out_path}  ({chk.raw_image_visible.shape}, bl={chk.black_level_per_channel[0]}, '
              f'wp={chk.white_level}, mean={chk.raw_image_visible.mean():.0f})')
        chk.close()


if __name__ == '__main__':
    main()
