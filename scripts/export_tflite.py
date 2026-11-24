# utf-8
"""把 PNNP 训练出的 SID-UNet 去噪模型导出为 TFLite。

路径：PyTorch 权重 -> Keras 重建同构网络（NHWC）-> TFLite
（不走 ONNX，避免额外依赖；权重逐层迁移后做数值校验）

用法：
  python3 scripts/export_tflite.py \
      --ckpt checkpoints/PNC460/PNC460_on_CombGT_Kcal_Unet_best_model.pth \
      --out  deploy/pnc460 \
      --size 512
产出：
  <out>/unet_<size>_fp32.tflite / _fp16.tflite / _int8.tflite
  <out>/export_report.txt
"""
import os, argparse, time
import numpy as np
import torch
import tensorflow as tf
from tensorflow import keras

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archs.Unet import UNetSeeInDark


def build_keras_unet(h, w=None, in_nc=4, out_nc=4, nf=32):
    """与 archs/Unet.UNetSeeInDark 完全同构（NHWC）。h,w 需能被 16 整除。"""
    if w is None:
        w = h
    L = keras.layers
    inp = keras.Input(shape=(h, w, in_nc), name='input')

    def conv(x, f, k=3, name=''):
        return L.Conv2D(f, k, strides=1, padding='same', use_bias=True, name=name)(x)

    def relu(x, name=''):
        return L.LeakyReLU(alpha=0.2, name=name)(x)

    def pool(x, name=''):
        return L.MaxPooling2D(2, strides=2, padding='valid', name=name)(x)

    def up(x, f, name=''):
        # PyTorch ConvTranspose2d(k=2, stride=2, padding=0) == TF padding='valid'
        return L.Conv2DTranspose(f, 2, strides=2, padding='valid', use_bias=True, name=name)(x)

    c1 = relu(conv(relu(conv(inp, nf, name='conv1_1'), name='relu1_1'), nf, name='conv1_2'), 'relu1_2')
    p1 = pool(c1, 'pool1')
    c2 = relu(conv(relu(conv(p1, nf*2, name='conv2_1'), name='relu2_1'), nf*2, name='conv2_2'), 'relu2_2')
    p2 = pool(c2, 'pool2')
    c3 = relu(conv(relu(conv(p2, nf*4, name='conv3_1'), name='relu3_1'), nf*4, name='conv3_2'), 'relu3_2')
    p3 = pool(c3, 'pool3')
    c4 = relu(conv(relu(conv(p3, nf*8, name='conv4_1'), name='relu4_1'), nf*8, name='conv4_2'), 'relu4_2')
    p4 = pool(c4, 'pool4')
    c5 = relu(conv(relu(conv(p4, nf*16, name='conv5_1'), name='relu5_1'), nf*16, name='conv5_2'), 'relu5_2')

    u6 = L.Concatenate(axis=-1, name='cat6')([up(c5, nf*8, 'upv6'), c4])
    c6 = relu(conv(relu(conv(u6, nf*8, name='conv6_1'), name='relu6_1'), nf*8, name='conv6_2'), 'relu6_2')
    u7 = L.Concatenate(axis=-1, name='cat7')([up(c6, nf*4, 'upv7'), c3])
    c7 = relu(conv(relu(conv(u7, nf*4, name='conv7_1'), name='relu7_1'), nf*4, name='conv7_2'), 'relu7_2')
    u8 = L.Concatenate(axis=-1, name='cat8')([up(c7, nf*2, 'upv8'), c2])
    c8 = relu(conv(relu(conv(u8, nf*2, name='conv8_1'), name='relu8_1'), nf*2, name='conv8_2'), 'relu8_2')
    u9 = L.Concatenate(axis=-1, name='cat9')([up(c8, nf, 'upv9'), c1])
    c9 = relu(conv(relu(conv(u9, nf, name='conv9_1'), name='relu9_1'), nf, name='conv9_2'), 'relu9_2')

    out = L.Conv2D(out_nc, 1, strides=1, padding='valid', use_bias=True, name='conv10_1')(c9)
    return keras.Model(inp, out, name='UNetSeeInDark')


def transfer_weights(pt_sd, kmodel):
    """PyTorch (NCHW) -> Keras (NHWC) 权重复制。conv/convT 权重都需 (2,3,1,0) 转置。"""
    km = {l.name: l for l in kmodel.layers}
    conv_names = [f'conv{i}_{j}' for i in range(1, 10) for j in (1, 2)] + ['conv10_1']
    up_names = [f'upv{i}' for i in range(6, 10)]
    missing = []
    for n in conv_names + up_names:
        if n not in km:
            missing.append(n); continue
        w = pt_sd[f'{n}.weight'].numpy()
        b = pt_sd[f'{n}.bias'].numpy()
        # Conv2d: (out,in,kh,kw) -> (kh,kw,in,out)
        # ConvTranspose2d: (in,out,kh,kw) -> (kh,kw,out,in)
        km[n].set_weights([w.transpose(2, 3, 1, 0), b])
    if missing:
        raise RuntimeError(f'缺少层: {missing}')


def representative_dataset(h, w, n=120, camera='PNC460'):
    """用项目真实的噪声合成分布做 int8 校准（而不是随机均匀噪声）。"""
    import rawpy, glob
    from utils import raw2bayer
    from archs.PNNP import PNNP
    from data_process.proxy import sample_params_proxy, generate_noisy_proxy
    from utils import load_weights, dataload

    WP, BL = 1023, 64
    pa = {'name': 'pw', 'noise_code': 'pr', 'ds_dir': '/root/chenbf/PNC460/resources',
          'ISO2K': [0.0158, 0], 'H': 3072, 'W': 4096, 'in_nc': 1, 'out_nc': 1,
          'nf': 16, 'nb': 2, 'nframes': 1, 'd': 1024, 'mode': '2stage+iso', 'act': 'swish'}
    net = PNNP(pa)
    net = load_weights(net, torch.load('/root/chenbf/pnnp_out/checkpoints/PNC460/'
                                       'pw_iso_pnc460_lowISO.pth',
                                       map_location='cpu', weights_only=True), by_name=True)
    net.ds_k = torch.from_numpy(dataload('/root/chenbf/PNC460/resources/darkshading_lowISO_k.npy').astype(np.float32))
    net.ds_b = torch.from_numpy(dataload('/root/chenbf/PNC460/resources/darkshading_lowISO_b.npy').astype(np.float32))
    net.pack_ds(); net.eval()

    # 校准样本：尺寸能放进真实 PNC460 帧时直接用真实噪声 raw（最贴近部署分布），
    # 否则用 Sony GT 合成
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real = sorted(glob.glob(os.path.join(REPO, 'raw_snap/lux*/SNAP_*.dng')))[:6]
    use_real = (h, w) == (1536, 2048) and len(real) > 0
    if use_real:
        samples = []
        for pth in real:
            rr = rawpy.imread(pth)
            pk = raw2bayer(rr.raw_image_visible.astype(np.float32), norm=False)
            x = np.clip((pk - BL) / (WP - BL), 0, 1) * 60.0
            x = np.clip(x, 0, 1)
            samples.append(torch.from_numpy(x))          # (4,h,w)
        def gen_real():
            for i in range(n):
                yield [samples[i % len(samples)].numpy().transpose(1, 2, 0)[None].astype(np.float32)]
        return gen_real

    gts = sorted(glob.glob('/root/chenbf/SID/Sony/long/0*.ARW'))[:20]
    patches = []
    for p in gts:
        raw = np.array(rawpy.imread(p).raw_image_visible).reshape(2848, 4256).astype(np.float32)
        pk = raw2bayer(raw, wp=16383, bl=512, norm=True, clip=True)[:, 800:800+h, 800:800+w]
        patches.append(torch.from_numpy(pk))  # (4,h,w)，与训练时 imgs_lr[i] 一致
    rng = np.random.default_rng(0)
    def gen():
        for i in range(n):
            gt = patches[i % len(patches)]
            prm = sample_params_proxy(camera_type=camera)
            prm['ISO'] = float(np.clip(prm['ISO'], 100, 1550))
            prm['shape'] = (4, 1, h, w); prm['device'] = 'cpu'
            with torch.no_grad():
                ny = generate_noisy_proxy(net, y=gt, param=prm, noise_code='pr',
                                          ori=False, camera_type=camera)
            arr = np.clip(ny.numpy(), 0, 1).transpose(1, 2, 0)[None]  # (4,h,w)->(1,h,w,4)
            yield [arr.astype(np.float32)]
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/root/chenbf/pnnp_out/checkpoints/PNC460/'
                                      'PNC460_on_CombGT_Kcal_Unet_best_model.pth')
    ap.add_argument('--out', default='/filesdir/chenbf/PNNP/deploy/pnc460')
    ap.add_argument('--size', type=int, default=512, help='方形尺寸（--height/--width 优先）')
    ap.add_argument('--height', type=int, default=0)
    ap.add_argument('--width', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    H = args.height or args.size
    W = args.width or args.size
    assert H % 16 == 0 and W % 16 == 0, 'H/W 必须能被 16 整除（4 次下采样）'
    TAG = f'{H}x{W}'
    report = []

    # ---- 1) 载入 PyTorch 权重 ----
    pt = UNetSeeInDark({'in_nc': 4, 'out_nc': 4, 'nf': 32, 'nframes': 1, 'use_dpsv': False,
                        'res': False, 'cascade': False, 'add': False, 'lock_wb': False})
    pt.load_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=True))
    pt.eval()
    nparam = sum(p.numel() for p in pt.parameters())
    report.append(f'参数量: {nparam:,}  ({nparam*4/1e6:.1f} MB fp32)')

    # ---- 2) Keras 重建 + 权重迁移 ----
    km = build_keras_unet(H, W)
    transfer_weights(pt.state_dict(), km)
    report.append(f'Keras 模型输入 (1,{H},{W},4)，层数 {len(km.layers)}')

    # ---- 3) 数值校验 ----
    x = np.random.default_rng(0).random((1, 4, H, W), dtype=np.float32)
    with torch.no_grad():
        y_pt = pt(torch.from_numpy(x)).numpy()                      # (1,4,H,W)
    y_tf = km(x.transpose(0, 2, 3, 1))                              # (1,H,W,4)
    y_tf = y_tf.numpy().transpose(0, 3, 1, 2)                       # (1,4,H,W)
    max_diff = float(np.abs(y_pt - y_tf).max())
    report.append(f'PyTorch vs Keras 最大误差: {max_diff:.2e}')
    assert max_diff < 1e-4, f'权重迁移误差过大: {max_diff}'

    def save(model_bytes, name):
        p = os.path.join(args.out, name)
        with open(p, 'wb') as f:
            f.write(model_bytes)
        report.append(f'{name}: {len(model_bytes)/1e6:.2f} MB')

    # ---- 4) float32 ----
    conv = tf.lite.TFLiteConverter.from_keras_model(km)
    save(conv.convert(), f'unet_{TAG}_fp32.tflite')

    # ---- 5) float16（权重半精度，速度/体积折中，精度几乎无损）----
    conv = tf.lite.TFLiteConverter.from_keras_model(km)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    save(conv.convert(), f'unet_{TAG}_fp16.tflite')

    # ---- 6) int8（真实噪声分布校准）----
    conv = tf.lite.TFLiteConverter.from_keras_model(km)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset(H, W)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.float32     # 保持 float I/O，内部 int8
    conv.inference_output_type = tf.float32
    save(conv.convert(), f'unet_{TAG}_int8.tflite')

    # ---- 7) 精度与速度实测 ----
    report.append('')
    report.append('量化精度对比（同一输入，相对 fp32 的 PSNR）:')
    ref = y_tf
    for tag in ['fp32', 'fp16', 'int8']:
        p = os.path.join(args.out, f'unet_{TAG}_{tag}.tflite')
        it = tf.lite.Interpreter(model_path=p, num_threads=4)
        it.allocate_tensors()
        inp = it.get_input_details()[0]; outd = it.get_output_details()[0]
        xin = x.transpose(0, 2, 3, 1).astype(np.float32)
        it.set_tensor(inp['index'], xin)
        t0 = time.time(); it.invoke(); dt = time.time() - t0
        y = it.get_tensor(outd['index']).transpose(0, 3, 1, 2)
        mse = float(((y - ref) ** 2).mean())
        psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
        report.append(f'  {tag:5s}: 与 fp32 差异 PSNR={psnr:6.2f} dB   本机 CPU 单次 {dt*1000:6.1f} ms '
                      f'({H}x{W})')

    txt = '\n'.join(report)
    print(txt)
    with open(os.path.join(args.out, 'export_report.txt'), 'w') as f:
        f.write(txt + '\n')
    print(f'\n输出目录: {args.out}')


if __name__ == '__main__':
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    main()
