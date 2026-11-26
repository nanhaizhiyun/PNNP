# utf-8
"""三方对比：C 版 / SNR 版 / SID 原模型。

对所有模型使用完全相同的推理后处理链（高光软削顶、亮像素旁路、
灰世界白平衡、显示域色度抑制、自适应色调映射），只换网络权重。

用法：
  python3 scripts/compare_models.py --out /root/chenbf/pnnp_out/compare_c_snr_sid
"""
import os, sys, argparse, glob, time
import numpy as np
import cv2
import rawpy
import torch
import importlib.util

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from archs.Unet import UNetSeeInDark
from utils import load_weights, raw2bayer

# 复用推理脚本里的后处理函数，保证三方完全同口径
spec = importlib.util.spec_from_file_location('inf', os.path.join(REPO, 'scripts/inference_pnc460.py'))
inf = importlib.util.module_from_spec(spec); spec.loader.exec_module(inf)

WP, BL = 1023, 64
BASE_ARGS = {'in_nc': 4, 'out_nc': 4, 'nf': 32, 'nframes': 1, 'use_dpsv': False,
             'res': False, 'cascade': False, 'add': False, 'lock_wb': False}

MODELS = {
    'C(ratio100-300)': ('torch', '/root/chenbf/pnnp_out/checkpoints/PNC460/'
                                 'PNC460_on_CombGT_Kcal_Unet_best_model.pth'),
    'SNR(ratio20-80)': ('torch', '/root/chenbf/pnnp_out/checkpoints/PNC460/'
                                 'PNC460_on_CombGT_SNR_Unet_best_model.pth'),
    'SID原模型':        ('sid',   '/root/chenbf/sid_pretrained/checkpoints/'
                                 'sid-ours-inc4/model_200_00257600.pt'),
}


def load_model(kind, path):
    net = UNetSeeInDark(BASE_ARGS)
    if kind == 'sid':
        sd = torch.load(path, map_location='cpu', weights_only=False)['netG']
    else:
        sd = torch.load(path, map_location='cpu', weights_only=True)
    return load_weights(net, sd, by_name=True).cuda().eval()


def infer(net, dng, ratio_fixed=None):
    """返回 (输出 packed 4 通道, 白平衡, 原始均值/标准差信息)"""
    r = rawpy.imread(dng)
    raw = r.raw_image_visible.astype(np.float32)
    wb = np.array(r.camera_whitebalance, dtype=np.float64); wb /= wb[1]
    x = np.clip((raw2bayer(raw, norm=False) - BL) / (WP - BL), 0, 1)

    if ratio_fixed:
        ratio = float(ratio_fixed)
        info = f'ratio={ratio:.0f}(固定)'
    else:
        ratio, info = inf.compute_gain(x)

    xin = x * ratio
    xn = inf.soft_clip_highlights(xin)
    with torch.no_grad():
        pred = net(torch.from_numpy(xn)[None].cuda()).clamp(0, 1)[0].cpu().numpy()
    pred = np.where(xin > 0.85, np.clip(xin, 0, 1), pred)      # 亮像素旁路

    # 统一后处理：灰世界 → 色调映射 → 显示域色度抑制
    r_ch = float(pred[0].mean()); g_ch = float((pred[1].mean() + pred[3].mean()) / 2)
    b_ch = float(pred[2].mean())
    wb_show = np.array([g_ch / max(r_ch, 1e-4), 1.0, g_ch / max(b_ch, 1e-4)])

    g = inf.auto_gamma(pred)
    rgb = inf.demosaic(pred, wb_show)
    rgb = np.clip(rgb, 0, 1) ** g
    chroma = inf.auto_chroma_strength(float(raw.std()))
    if chroma < 1.0:
        rgb = inf.chroma_denoise_display(rgb, chroma)
    return pred, rgb, info


def metrics(rgb):
    """细节能量 / 残余噪声 / 噪声比 / 色度噪声"""
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    detail = float(cv2.Laplacian(gray, cv2.CV_32F).std())
    from scipy.ndimage import uniform_filter
    loc = uniform_filter(gray, 9)
    res = gray - loc
    gy, gx = np.gradient(loc.astype(np.float32))
    flat = (np.abs(gy) + np.abs(gx)) < np.percentile(np.abs(gy) + np.abs(gx), 50)
    noise = float(res[flat].std()) if flat.sum() > 1000 else float('nan')
    lab = cv2.cvtColor(np.uint8(np.clip(rgb, 0, 1) * 255), cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = float((lab[..., 1] - 128).std() + (lab[..., 2] - 128).std())
    return {'detail': detail, 'noise': noise, 'noise_ratio': noise / max(detail, 1e-6),
            'chroma': chroma}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/root/chenbf/pnnp_out/compare_c_snr_sid')
    ap.add_argument('--raw-snap-ratio', type=float, default=60.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    nets = {k: load_model(*v) for k, v in MODELS.items()}
    print('模型加载完成:', list(nets))

    datasets = {
        'test_pnc460': (sorted(glob.glob('/filesdir/chenbf/test_pnc460/*.dng')), None),
        'raw_snap_lux0.2': (sorted(glob.glob(os.path.join(
            REPO, 'raw_snap/lux0.2/SNAP_*.dng')))[:1], args.raw_snap_ratio),
        'raw_snap_lux0.5': (sorted(glob.glob(os.path.join(
            REPO, 'raw_snap/lux0.5/SNAP_*.dng')))[:1], args.raw_snap_ratio),
    }

    rows = []
    for dname, (files, ratio) in datasets.items():
        if not files:
            print(f'[跳过] {dname}: 无文件'); continue
        for tag, net in nets.items():
            per = {k: [] for k in ('detail', 'noise', 'noise_ratio', 'chroma')}
            t0 = time.time()
            for f in files:
                _, rgb, info = infer(net, f, ratio)
                m = metrics(rgb)
                for k in per: per[k].append(m[k])
                if len(files) < 5:   # 少量图时保存可视化
                    sub = os.path.join(args.out, dname)
                    os.makedirs(sub, exist_ok=True)
                    cv2.imwrite(os.path.join(sub, f'{tag}_{os.path.basename(f)[:-4]}.png'),
                                np.uint8(np.clip(rgb, 0, 1)[..., ::-1] * 255))
            row = {'dataset': dname, 'model': tag,
                   **{k: float(np.nanmean(v)) for k, v in per.items()},
                   'sec': (time.time() - t0) / len(files)}
            rows.append(row)
            print(f"{dname:18s} {tag:16s} detail={row['detail']:.4f} noise={row['noise']:.4f} "
                  f"noise/detail={row['noise_ratio']:.2f} chroma={row['chroma']:.2f} "
                  f"({row['sec']:.1f}s/张)")

    # 汇总表
    lines = []
    cur = None
    for r in rows:
        if r['dataset'] != cur:
            cur = r['dataset']
            lines.append(f"\n=== {cur} ===")
            lines.append(f"{'模型':<18}{'细节能量':>10}{'残余噪声':>10}{'噪声/细节':>10}{'色度噪声':>10}")
        lines.append(f"{r['model']:<18}{r['detail']:>10.4f}{r['noise']:>10.4f}"
                     f"{r['noise_ratio']:>10.2f}{r['chroma']:>10.2f}")
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
        f.write(txt + '\n')
    print(f'\n输出: {args.out}')


if __name__ == '__main__':
    main()
