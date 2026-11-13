# utf-8
"""暗帧数据集：为 PNNP 本体训练提供像素级残差噪声样本与标定产物。

数据组织约定（与 SFRN_Dataset 一致）：
    bias_dir/<ISO>/*.ARW 或 *.npy（npy 便于合成数据测试/无 rawpy 环境）

标定（对应论文 PND 解耦的前两级）：
    对每个 ISO：mean_frame = 多帧均值（含 band-wise FPN + frame-wise BLE 的恒定部分）
    训练目标：residual = frame - mean_frame（可选再去每帧全局均值，保证 i.i.d. 前提）
    标定产物：BLE[iso]（逐 ISO 标量）+ 逐像素线性拟合 k*iso + b（与官方 resources 格式一致）
"""
import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import dataload, raw2bayer, log


class DarkFrameDataset(Dataset):
    """按 ISO 懒加载暗帧，采样像素级残差噪声 patch。"""

    def __init__(self, args=None):
        self.default_args()
        if args is not None:
            for key in args:
                self.args[key] = args[key]
        bias_dir = self.args['bias_dir']
        # 与 SFRN_Dataset 相同的目录约定：子目录名为 ISO
        iso_dirs = sorted(os.listdir(bias_dir), key=lambda x: int(x))
        self.isos = [int(d) for d in iso_dirs]
        self.files = {iso: sorted(glob.glob(os.path.join(bias_dir, d, '*')))
                      for iso, d in zip(self.isos, iso_dirs)}
        if self.args['max_frames_per_iso'] > 0:
            self.files = {iso: fs[:self.args['max_frames_per_iso']] for iso, fs in self.files.items()}
        for iso in self.isos:
            assert len(self.files[iso]) >= 2, f'ISO {iso} 暗帧不足 2 张，无法估计残差'

        # 逐 ISO 的统计量与残差（load_iso 时填充，全部常驻内存以支持快速联合采样）
        self.mean_frames = {}      # iso -> (4, h, w) float32 均值帧（bayer packed）
        self.BLE = {}              # iso -> float 全局均值（DN）
        self._res_all = {}         # iso -> (N, 4, h, w) float32 残差
        self.hot_masks = {}        # iso -> (4, h, w) bool 热像素掩膜（True=热像素）
        self.hot_ratios = {}       # iso -> float 热像素占比
        self._cache_iso = None
        self._cache_res = None
        log(f'DarkFrameDataset: ISO list = {self.isos}')

    def default_args(self):
        self.args = {}
        self.args['bias_dir'] = None
        self.args['H'] = 2848
        self.args['W'] = 4256
        self.args['wp'] = 16383
        self.args['bl'] = 512
        self.args['patch_size'] = 1024
        self.args['remove_frame_mean'] = True
        self.args['max_frames_per_iso'] = 0
        self.args['hot_pixel_k'] = 10.0     # 时间std > k*中位数 判定为热像素；0=关闭

    def load_iso(self, iso):
        """加载某个 ISO 的全部暗帧并计算均值帧与残差（常驻缓存，重复调用直接命中）。"""
        if iso in self._res_all:
            self._cache_iso = iso
            self._cache_res = self._res_all[iso]
            return
        H, W = self.args['H'], self.args['W']
        stack = []
        for path in tqdm(self.files[iso], desc=f'Load dark frames ISO{iso}'):
            raw = np.array(dataload(path)).reshape(H, W).astype(np.float32)
            stack.append(raw2bayer(raw, norm=False))  # (4, h, w) DN
        stack = np.stack(stack, axis=0)  # (N, 4, h, w)
        mean_frame = stack.mean(axis=0)
        res = stack - mean_frame
        if self.args['remove_frame_mean']:
            res = res - res.mean(axis=(2, 3), keepdims=True)
        # 样本均值收缩修正：residual = x_i - x̄ 的方差为 σ²(1-1/N)，
        # 在 i.i.d. 假设下乘 1/sqrt(1-1/N) 还原真实噪声标准差（N 小时间题显著，如 N=5 收缩 10.6%）
        N = stack.shape[0]
        res = res * (1.0 / np.sqrt(1.0 - 1.0 / N))
        # 热像素检测：暗帧中位置固定、时间波动远大于普通读噪声的像素，
        # 破坏 i.i.d. 假设且显著拉高 std/尾部，需掩膜剔除
        k = self.args.get('hot_pixel_k', 0.0)
        if k > 0:
            tstd = res.std(axis=0)                      # (4, h, w) 逐像素时间std
            med = np.median(tstd)
            hot = tstd > k * med
        else:
            hot = np.zeros(res.shape[1:], dtype=bool)
        self.hot_masks[iso] = hot
        self.hot_ratios[iso] = float(hot.mean())
        log(f'ISO{iso}: 热像素 {hot.sum()} px ({self.hot_ratios[iso]*100:.4f}%, k={k})')
        self.mean_frames[iso] = mean_frame
        self.BLE[iso] = float(mean_frame.mean())
        self._res_all[iso] = res.astype(np.float32)
        self._cache_iso = iso
        self._cache_res = self._res_all[iso]

    def sample_residual(self, iso, patch_size=None):
        """随机取一张暗帧的残差 patch，返回 (4, 1, ps, ps) float32 tensor（DN 单位）。"""
        crop, _ = self.sample_pair(iso, patch_size)
        return crop

    def sample_pair(self, iso, patch_size=None):
        """同 sample_residual，但额外返回热像素掩膜 patch（True=热像素）。

        返回 (crop, mask)，均为 (4, 1, ps, ps)；未启用热像素检测时 mask 全 False。
        """
        self.load_iso(iso)
        ps = patch_size or self.args['patch_size']
        res = self._cache_res  # (N, 4, h, w)
        N, _, h, w = res.shape
        assert ps <= h and ps <= w, f'patch_size {ps} 超过暗帧 packed 尺寸 {h}x{w}'
        i = np.random.randint(N)
        x = np.random.randint(h - ps + 1)
        y = np.random.randint(w - ps + 1)
        crop = res[i, :, x:x + ps, y:y + ps]
        mask = self.hot_masks[iso][:, x:x + ps, y:y + ps]
        return (torch.from_numpy(crop).unsqueeze(1),
                torch.from_numpy(mask).unsqueeze(1))

    def calibrate(self, isos, ds_dir, tag='lowISO'):
        """拟合 darkshading 线性模型并保存（与官方 resources 格式兼容）。

        mean_frame(iso) ≈ k * iso + b + BLE[iso]
        产物：darkshading_BLE.pkl（dict: iso -> float，含全部已加载 ISO）
              darkshading_{tag}_k.npy / _b.npy（float64, H x W 未打包格式）
        """
        os.makedirs(ds_dir, exist_ok=True)
        isos = sorted(isos)
        for iso in isos:
            self.load_iso(iso)
        H, W = self.args['H'], self.args['W']
        # 联立所有 ISO 逐像素最小二乘: D(iso) ≈ k*iso + b
        A = np.stack([np.array(isos, dtype=np.float64), np.ones(len(isos))], axis=1)  # (n_iso, 2)
        M = np.stack([self._fill_hot(self.mean_frames[iso], self.hot_masks.get(iso))
                      - self.BLE[iso] for iso in isos])  # (n_iso, 4, h, w)
        coef = np.linalg.pinv(A) @ M.reshape(len(isos), -1)  # (2, 4*h*w)
        k_map = self._unpack(coef[0].reshape(4, H // 2, W // 2).astype(np.float64), H, W)
        b_map = self._unpack(coef[1].reshape(4, H // 2, W // 2).astype(np.float64), H, W)
        np.save(os.path.join(ds_dir, f'darkshading_{tag}_k.npy'), k_map)
        np.save(os.path.join(ds_dir, f'darkshading_{tag}_b.npy'), b_map)
        ble_path = os.path.join(ds_dir, 'darkshading_BLE.pkl')
        ble = {}
        if os.path.exists(ble_path):
            with open(ble_path, 'rb') as f:
                ble = pickle.load(f)
        ble.update(self.BLE)
        with open(ble_path, 'wb') as f:
            pickle.dump(ble, f)
        log(f'Calibration saved to {ds_dir}: k/b ({tag}), BLE ({len(ble)} ISOs)')

    @staticmethod
    def _fill_hot(mean_frame, hot):
        """均值帧中热像素位置用 5x5 中值滤波值填充（未掩膜时原样返回）。"""
        if hot is None or not hot.any():
            return mean_frame
        from scipy.ndimage import median_filter
        filled = median_filter(mean_frame, size=(1, 5, 5))
        return np.where(hot, filled, mean_frame)

    @staticmethod
    def _unpack(packed, H, W):
        """(4, h, w) bayer packed -> (H, W)，raw2bayer 的逆变换。"""
        raw = np.empty((H, W), dtype=packed.dtype)
        raw[0:H:2, 0:W:2] = packed[0]
        raw[0:H:2, 1:W:2] = packed[1]
        raw[1:H:2, 1:W:2] = packed[2]
        raw[1:H:2, 0:W:2] = packed[3]
        return raw

    def __len__(self):
        return len(self.isos)

    def __getitem__(self, idx):
        raise NotImplementedError('请使用 sample_residual(iso, patch_size) 采样')
