# utf-8
"""PNC460 去噪推理脚本（部署版）。

增益策略（相比均值版的两处改进）：
  1. 分位数增益：用 99% 分位数代替均值估计亮度——"大部分暗+局部亮"的
     双峰场景不再被暗部绑架而把高光推爆。
  2. HDR 高光保护：估计增益后检测将饱和的像素比例，超过阈值则迭代回退
     增益（每次 ×0.7），宁可暗部欠曝也不让高光硬削顶产生伪影。

用法：
  python3 scripts/inference_pnc460.py --input <目录或文件> --output <目录> \
      --model checkpoints/PNC460/PNC460_on_CombGT_Kcal_Unet_best_model.pth
"""
import os
import argparse
import numpy as np
import cv2
import rawpy
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archs.Unet import UNetSeeInDark
from utils import load_weights, raw2bayer

WP, BL = 1023, 64          # PNC460 白/黑电平


def pack2rgb(p4, wb, gamma=None):
    """packed bayer (4,h,w) -> sRGB（双线性去马赛克 + 白平衡 + gamma，演示级）。

    gamma=None 时用 1/2.2 标准曲线；否则用给定值（<1 提亮暗部）。
    """
    if gamma is None:
        gamma = 1 / 2.2
    r = p4[0]; g = (p4[1] + p4[3]) / 2; b = p4[2]
    h, w = r.shape
    rgb = np.zeros((h, w, 3))
    rgb[0::2, 0::2] = np.stack([r[0::2, 0::2], g[0::2, 0::2], b[0::2, 0::2]], -1)
    rgb[0::2, 1::2] = np.stack([r[0::2, 1::2], g[0::2, 1::2], b[0::2, 1::2]], -1)
    rgb[1::2, 0::2] = np.stack([r[1::2, 0::2], g[1::2, 0::2], b[1::2, 0::2]], -1)
    rgb[1::2, 1::2] = np.stack([r[1::2, 1::2], g[1::2, 1::2], b[1::2, 1::2]], -1)
    rgb = cv2.resize(rgb, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    rgb *= np.array([wb[0], 1.0, wb[2]])
    return np.clip(rgb, 0, 1) ** gamma


def auto_gamma(img01, target_mean=0.42):
    """自适应 gamma：按输出亮度估计提亮系数，暗图变亮、亮图不变。

    高光保护会把整体压暗（正确做法，线性域保住了高光），显示阶段
    用 gamma 提暗部——分工同真实 ISP 的 tone mapping。
    """
    mean = float(img01.mean())
    if mean <= 0: return 1 / 2.2
    gamma = np.log(target_mean) / np.log(mean)
    return float(np.clip(gamma, 0.35, 1.0))


def soft_clip_highlights(x, knee=0.75, top=0.97):
    """输入高光软削顶：硬削顶的边缘（某像素 1.0、邻居 0.3）会诱发网络
    过冲/色边（B/C 模型训练集中不存在该状态）。用 tanh 在 knee 以上平滑
    压缩到 top，消除硬边。线性区（<knee）不受影响。"""
    out = x.copy()
    mask = x > knee
    out[mask] = knee + (top - knee) * np.tanh((x[mask] - knee) / (top - knee))
    return out


def compute_gain(x, target=0.12, p=99.0, sat_limit=0.95, sat_frac_max=0.02,
                 ratio_min=2.0, ratio_max=300.0):
    """分位数增益 + HDR 高光保护。返回 (ratio, 信息串)。

    ratio_min 取 2 而非历史值 20：训练 ratio 范围(100-300)只是合成分布，
    推理时宁可低增益也不能让高光硬削顶——模型对低 ratio 输入的泛化
    已在本项目 test_pnc460(ratio 25-45)验证无碍。"""
    p_v = float(np.percentile(x, p))
    ratio = float(np.clip(target / max(p_v, 1e-5), ratio_min, ratio_max))
    reason = f'p{p}={p_v:.4f}'
    # HDR 高光保护：饱和像素比例超限则回退
    while ratio > ratio_min:
        sat_frac = float((x * ratio > sat_limit).mean())
        if sat_frac <= sat_frac_max:
            reason += f' 饱和={sat_frac*100:.1f}%'
            break
        ratio = max(ratio * 0.7, ratio_min)
        if ratio == ratio_min:
            reason += f' 已达增益下限(饱和{sat_frac*100:.1f}%)'
    return ratio, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', '-i', required=True, help='DNG 文件或目录')
    ap.add_argument('--output', '-o', required=True, help='输出目录')
    ap.add_argument('--model', '-m', default='/root/chenbf/pnnp_out/checkpoints/PNC460/'
                                             'PNC460_on_CombGT_Kcal_Unet_best_model.pth')
    ap.add_argument('--target', '-t', type=float, default=0.12, help='增益目标（分位数值）')
    ap.add_argument('--no-signal-guard', action='store_true', help='关闭零信号护栏（默认开启）')
    ap.add_argument('--no-soft-clip', action='store_true', help='关闭输入高光软削顶（默认开启）')
    ap.add_argument('--bypass-thr', type=float, default=0.85,
                    help='高光旁路阈值：输入(增益后)高于此值的像素直接透传，设 1.0 关闭')
    args = ap.parse_args()

    net = load_weights(
        UNetSeeInDark({'in_nc': 4, 'out_nc': 4, 'nf': 32, 'nframes': 1, 'use_dpsv': False,
                       'res': False, 'cascade': False, 'add': False, 'lock_wb': False}),
        torch.load(args.model, map_location='cpu', weights_only=True), by_name=True).cuda().eval()

    if os.path.isdir(args.input):
        files = sorted(f for f in os.listdir(args.input) if f.endswith('.dng'))
        paths = [os.path.join(args.input, f) for f in files]
    else:
        paths = [args.input]
    os.makedirs(args.output, exist_ok=True)

    for path in paths:
        name = os.path.basename(path)[:-4]
        r = rawpy.imread(path)
        raw = r.raw_image_visible.astype(np.float32)
        wb = np.array(r.camera_whitebalance, dtype=np.float64); wb = wb / wb[1]
        x = np.clip((raw2bayer(raw, norm=False) - BL) / (WP - BL), 0, 1)

        # 零信号护栏：raw 空间 std 接近暗帧噪声水平 = 光学全黑
        raw_std = float(raw.std())
        if not args.no_signal_guard and raw_std < 5.0:
            print(f'{name}: [护栏拦截] raw std={raw_std:.1f} DN ≈ 纯噪声，'
                  f'跳过处理（请确认镜头有光进入）。可用 --no-signal-guard 强制处理')
            continue

        ratio, info = compute_gain(x, target=args.target)
        xin = x * ratio
        # 高光补丁1：输入软削顶，消除硬削顶边缘（网络在该状态无训练覆盖）
        xin_net = soft_clip_highlights(xin) if not args.no_soft_clip else np.clip(xin, 0, 1)
        with torch.no_grad():
            pred = net(torch.from_numpy(xin_net)[None].cuda()).clamp(0, 1)[0].cpu().numpy()
        # 高光补丁2：亮像素旁路——足够亮的区域信噪比高、噪声不可见，
        # 直接透传输入，不经网络（避免模型在高光区的过冲幻觉）
        if args.bypass_thr < 1.0:
            bright = xin > args.bypass_thr
            pred = np.where(bright, np.clip(xin, 0, 1), pred)
        g = auto_gamma(pred)
        out_png = np.uint8(np.clip(pack2rgb(pred, wb, gamma=g), 0, 1)[..., ::-1] * 255)
        cv2.imwrite(os.path.join(args.output, f'{name}_denoised.png'), out_png)
        # 同步保存干净 raw（DN uint16，供下游 ISP 使用）
        raw_out = np.clip(pred / max(ratio, 1e-6), 0, 1) * (WP - BL)
        np.save(os.path.join(args.output, f'{name}_clean_raw.npy'), raw_out.astype(np.uint16))
        print(f'{name}: ratio={ratio:.0f} ({info}) gamma={g:.2f} '
              f'旁路={float((xin > args.bypass_thr).mean()) * 100:.1f}% std={raw_std:.1f} -> 已保存')


if __name__ == '__main__':
    main()
