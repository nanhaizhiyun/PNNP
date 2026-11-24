# utf-8
"""在真实 PNC460 raw 上验证 TFLite 模型与 PyTorch 模型是否一致。

用法：
  python3 scripts/verify_tflite.py \
      --dng raw_snap/lux0.5/SNAP_ISO1550_ET99.77ms_000.dng \
      --tflite deploy/pnc460/unet_512_fp16.tflite \
      --ckpt checkpoints/PNC460/PNC460_on_CombGT_Kcal_Unet_best_model.pth \
      --ratio 60
"""
import os, argparse, sys
import numpy as np
import cv2
import rawpy
import torch
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archs.Unet import UNetSeeInDark
from utils import raw2bayer

WP, BL = 1023, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dng', required=True)
    ap.add_argument('--tflite', required=True)
    ap.add_argument('--ckpt', default='/root/chenbf/pnnp_out/checkpoints/PNC460/'
                                      'PNC460_on_CombGT_Kcal_Unet_best_model.pth')
    ap.add_argument('--ratio', type=float, default=60.0)
    ap.add_argument('--out', default='/root/chenbf/pnnp_out/tflite_verify')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- 输入：与推理脚本一致的前处理 ----
    r = rawpy.imread(args.dng)
    raw = r.raw_image_visible.astype(np.float32)
    x = np.clip((raw2bayer(raw, norm=False) - BL) / (WP - BL), 0, 1) * args.ratio
    knee, top = 0.75, 0.97
    m = x > knee
    x[m] = knee + (top - knee) * np.tanh((x[m] - knee) / (top - knee))

    size = 512
    ch, cw = x.shape[1] // 2, x.shape[2] // 2
    crop = np.ascontiguousarray(x[:, ch - size // 2:ch + size // 2,
                                   cw - size // 2:cw + size // 2])   # (4,S,S)

    # ---- PyTorch ----
    pt = UNetSeeInDark({'in_nc': 4, 'out_nc': 4, 'nf': 32, 'nframes': 1, 'use_dpsv': False,
                        'res': False, 'cascade': False, 'add': False, 'lock_wb': False})
    pt.load_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=True))
    pt.eval()
    with torch.no_grad():
        y_pt = pt(torch.from_numpy(crop)[None])[0].numpy()

    # ---- TFLite ----
    it = tf.lite.Interpreter(model_path=args.tflite, num_threads=4)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    outd = it.get_output_details()[0]
    print(f'TFLite 输入 {inp["shape"]} {inp["dtype"].__name__} | 输出 {outd["shape"]}')
    xin = crop.transpose(1, 2, 0)[None].astype(np.float32)      # (1,S,S,4)
    it.set_tensor(inp['index'], xin)
    t0 = __import__('time').time()
    it.invoke()
    dt = __import__('time').time() - t0
    y_tf = it.get_tensor(outd['index'])[0].transpose(2, 0, 1)    # (4,S,S)

    # ---- 对比 ----
    diff = np.abs(y_pt - y_tf)
    mse = float((diff ** 2).mean())
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    print(f'真实数据一致性: max|diff|={diff.max():.2e}  mean={diff.mean():.2e}  PSNR={psnr:.2f} dB')
    print(f'本机 CPU 单次推理: {dt*1000:.1f} ms ({size}x{size})')

    # ---- 可视化 ----
    def save(p4, wb, name):
        rr = p4[0]; g = (p4[1] + p4[3]) / 2; b = p4[2]
        h, w = rr.shape
        rgb = np.zeros((h, w, 3))
        rgb[0::2, 0::2] = np.stack([rr[0::2, 0::2], g[0::2, 0::2], b[0::2, 0::2]], -1)
        rgb[0::2, 1::2] = np.stack([rr[0::2, 1::2], g[0::2, 1::2], b[0::2, 1::2]], -1)
        rgb[1::2, 0::2] = np.stack([rr[1::2, 0::2], g[1::2, 0::2], b[1::2, 0::2]], -1)
        rgb[1::2, 1::2] = np.stack([rr[1::2, 1::2], g[1::2, 1::2], b[1::2, 1::2]], -1)
        rgb = cv2.resize(rgb, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        rgb *= np.array([wb[0], 1.0, wb[2]])
        v = np.clip(rgb, 0, 1) ** 0.5
        cv2.imwrite(f'{args.out}/{name}.png', np.uint8(v[..., ::-1] * 255))

    wb = np.array(r.camera_whitebalance, dtype=np.float64); wb /= wb[1]
    save(y_pt, wb, 'torch')
    save(y_tf, wb, 'tflite')
    save(np.clip(crop, 0, 1), wb, 'input')
    print(f'对比图 -> {args.out}/')


if __name__ == '__main__':
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    main()
