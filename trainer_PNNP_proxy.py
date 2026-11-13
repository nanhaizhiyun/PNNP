# utf-8
"""PNNP 本体训练器：从暗帧学习像素级噪声代理（对应论文 PPM + DDL）。

训练目标：使 proxy_pixel 生成的噪声分布与暗帧残差（去除 frame-wise BLE 与
band-wise dark shading 后的像素级 i.i.d. 噪声）分布一致。

损失（DDL, Differentiable Distribution Loss）：
    L = w_q * |F_out^{-1}(p_k) - F_real^{-1}(p_k)| + w_c * |F_out(q_k) - F_real(q_k)|
    p_k 为均匀概率网格（get_x 'uniform'），q_k 为标准正态分位点+高斯扰动（get_x 'icdf'）。
    损失作用在按真值 std 标准化后的样本上，使 N(0,1) 空间的查询点有效。

流程：ISO 按 iso_threshold 分成 low/high（双原生 ISO）两个分支，各训一个网络；
每个 ISO 训练 steps_per_iso 步（SGDR: lr_max -> lr_min），patch 随机裁剪。

导出：{fast_ckpt}/{model_name}_{branch}ISO.pth，键名与官方发布权重一致
（'proxy_pixel.*'），可直接被 trainer_PNNP_SID.load_proxynet 加载。

简化说明（与论文实现的差异）：
    1. 未实现 SFRN 的 high-bit 重建（量化失配校正），高 ISO 下残差远大于 1 DN，
       影响有限；低噪声场景可后续补充。
    2. 论文按"每 ISO 1000 步"训练；本实现以 cycles 参数支持多轮循环。
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from data_process import *
from utils import *
from archs import *
from archs.PNNP import PNNP, QuantileLoss, CDFLoss, get_x


class ProxyParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument('--runfile', '-f', default="runfiles/SonyA7S2/PNNP_proxy_train.yml", type=str, help="path to config")
        self.parser.add_argument('--gpu', default="0", help="os.environ['CUDA_VISIBLE_DEVICES']")
        return self.parser.parse_args()


class PNNP_Proxy_Trainer():
    def __init__(self):
        self.parser = ProxyParser().parse()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.parser.gpu
        with open(self.parser.runfile, 'r', encoding="utf-8") as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.arch_proxy = self.args['arch_proxy']
        self.hyper = self.args['hyper']
        self.dst = self.args['dst']
        self.model_name = self.args['model_name']
        self.fast_ckpt = self.args['fast_ckpt']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 路径解析：项目内 -> $PNNP_DATA_ROOT -> /data/fenghansen/datasets
        # bias_dir 是输入目录必须已存在；ds_dir 是输出目录，数据根存在即可
        for key in ['bias_dir', 'ds_dir']:
            path = self.dst[key]
            if not os.path.isabs(path) and not os.path.exists(os.path.abspath(path)):
                for root in filter(None, [os.environ.get('PNNP_DATA_ROOT'), '/data/fenghansen/datasets']):
                    cand = os.path.join(root, path)
                    if os.path.exists(cand) or (key == 'ds_dir' and os.path.isdir(root)):
                        path = cand
                        break
            self.dst[key] = path
        os.makedirs(self.fast_ckpt, exist_ok=True)
        os.makedirs('./logs', exist_ok=True)
        self.logfile = f'./logs/log_{self.model_name}_proxy.log'

        np.random.seed(self.hyper.get('seed', 0))
        torch.manual_seed(self.hyper.get('seed', 0))
        self.dstset = DarkFrameDataset(self.dst)
        self.q_points = self.hyper.get('query_points', 1000)
        self.x_quant = get_x(mode='uniform', size=self.q_points)
        self.x_cdf = get_x(mode='icdf', size=self.q_points)

    def build_proxy(self):
        net = PNNP(self.arch_proxy).to(self.device)
        net.ISO2K_require_grad(False, ISO2K=self.arch_proxy['ISO2K'])
        return net

    def ddl_loss(self, output, gt):
        """可微分分布损失：分位数损失 + CDF 损失（标准化到单位方差后计算）。

        实现说明：torch.quantile 在百万级样本×上千分位点下极慢（实测 65s/step），
        且论文明确说明"排序保持梯度可微"。因此单次 sort 后做插值，同时服务
        分位数损失（在概率位置插值 sorted 样本）与 CDF 损失（在数值点插值
        经验 CDF），与 archs/PNNP.py 的 QuantileLoss/CDFLoss 数学等价。
        """
        scale = gt.std().detach()
        out = (output / scale).reshape(-1)
        tar = (gt / scale).reshape(-1)
        s_out, _ = torch.sort(out)
        s_gt, _ = torch.sort(tar)
        n = s_out.numel()

        # 分位数损失: F^-1(p) = sorted 在位置 p*(n-1) 的线性插值
        pos = (self.x_quant.to(self.device) * (n - 1)).clamp(0, n - 1)
        i0 = pos.floor().long()
        w = pos - i0.float()
        i1 = (i0 + 1).clamp(max=n - 1)
        q_out = s_out[i0] * (1 - w) + s_out[i1] * w
        q_gt = s_gt[i0] * (1 - w) + s_gt[i1] * w
        lq = F.l1_loss(q_out, q_gt)

        # CDF 损失: F(q) = searchsorted + 线性插值
        def cdf_interp(s, x):
            idx = torch.searchsorted(s, x).clamp(1, n - 1)
            sl, sr = s[idx - 1], s[idx]
            frac = (x - sl) / (sr - sl).clamp_min(1e-12)
            return (idx - 1 + frac) / (n - 1)
        lc = F.l1_loss(cdf_interp(s_out, self.x_cdf.to(self.device)),
                       cdf_interp(s_gt, self.x_cdf.to(self.device)))
        return lq, lc

    def train_branch(self, branch, isos):
        """训练一个 ISO 分支（如 lowISO），逐 ISO 训练 steps_per_iso 步。"""
        log(f'==== Train branch {branch}: ISOs {isos} ====', log=self.logfile)
        net = self.build_proxy()
        # 预加载分支内全部 ISO 的暗帧残差（常驻内存，避免逐步骤换 ISO 反复读盘）
        for iso in isos:
            self.dstset.load_iso(iso)
        # 先保存该分支的标定产物（darkshading k/b + BLE）
        self.dstset.calibrate(isos, self.dst['ds_dir'], tag=branch)

        lr_max = self.hyper['learning_rate']
        lr_min = self.hyper['min_learning_rate']
        steps = self.hyper['steps_per_iso']
        ps = self.dst['patch_size']
        noise_code = self.arch_proxy['noise_code']

        for cycle in range(self.hyper.get('cycles', 1)):
            # 联合训练：每步把分支内所有 ISO 各采一个 patch，梯度累加后统一更新。
            # 逐 ISO 串行全参数训练会导致灾难性遗忘（网络只记住最后训练的 ISO），
            # K(ISO) 增益分解需要在所有 ISO 上同时约束才能解耦。
            optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr_max)
            # LambdaLR 以 base lr 为基准做乘法，lambda 需返回比例
            scheduler = LambdaLR(optimizer,
                                 lambda t: (lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / steps))) / lr_max)
            net.train()
            time_start = time.time()
            for step in range(1, steps + 1):
                loss_acc, lq_acc, lc_acc = 0.0, 0.0, 0.0
                out_std_log, gt_std_log = [], []
                optimizer.zero_grad()
                for iso in isos:
                    gt = self.dstset.sample_residual(iso, ps).to(self.device)
                    p = {'ISO': iso, 'shape': (4, 1, ps, ps), 'device': self.device, 'noise_code': noise_code}
                    output = net.proxy_pixel(p)
                    lq, lc = self.ddl_loss(output, gt)
                    loss = self.hyper.get('w_quantile', 1.0) * lq + self.hyper.get('w_cdf', 1.0) * lc
                    (loss / len(isos)).backward()  # 梯度累加（等价于平均损失）
                    loss_acc += loss.item() / len(isos)
                    lq_acc += lq.item() / len(isos)
                    lc_acc += lc.item() / len(isos)
                    out_std_log.append(f'{iso}:{output.std().item():.2f}')
                    gt_std_log.append(f'{iso}:{gt.std().item():.2f}')
                optimizer.step()
                scheduler.step()
                if step % self.hyper.get('log_freq', 100) == 0 or step == 1:
                    log(f'[{branch} c{cycle} step {step}/{steps}] '
                        f'loss={loss_acc:.4f} (q={lq_acc:.4f}, cdf={lc_acc:.4f}) '
                        f'out_std[{" ".join(out_std_log)}] gt_std[{" ".join(gt_std_log)}] '
                        f'lr={scheduler.get_last_lr()[0]:.2e} '
                        f'({(time.time() - time_start) / step:.2f}s/step)', log=self.logfile)
            # 分支训完后对所有 ISO 做分布校验（防止遗忘）
            net.eval()
            with torch.no_grad():
                for iso in isos:
                    p = {'ISO': iso, 'shape': (4, 1, ps, ps), 'device': self.device, 'noise_code': noise_code}
                    out = net.proxy_pixel(p).reshape(-1).cpu().numpy()
                    gt_all = self.dstset.sample_residual(iso, ps).reshape(-1).numpy()
                    log(f'[{branch} ISO{iso}] check: out_std={out.std():.3f} gt_std={gt_all.std():.3f} | '
                        f'out_q1/q99={np.quantile(out, 0.01):.2f}/{np.quantile(out, 0.99):.2f} '
                        f'gt_q1/q99={np.quantile(gt_all, 0.01):.2f}/{np.quantile(gt_all, 0.99):.2f}',
                        log=self.logfile)

        save_path = os.path.join(self.fast_ckpt, f'{self.model_name}_{branch}.pth')
        sd = {f'proxy_pixel.{k}': v for k, v in net.proxy_pixel.state_dict().items()}
        torch.save(sd, save_path)
        log(f'Branch {branch} saved: {save_path}', log=self.logfile)
        return net

    def train(self):
        isos = self.dstset.isos
        thr = self.dst.get('iso_threshold', 1600)
        branches = {'lowISO': [i for i in isos if i <= thr],
                    'highISO': [i for i in isos if i > thr]}
        for branch, b_isos in branches.items():
            if len(b_isos) == 0:
                log(f'Branch {branch}: no ISOs, skip', log=self.logfile)
                continue
            self.train_branch(branch, b_isos)
        log('PNNP proxy training finished.', log=self.logfile)


if __name__ == '__main__':
    trainer = PNNP_Proxy_Trainer()
    trainer.train()
