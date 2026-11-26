# utf-8
"""NAFNet (Simple Baselines for Image Restoration, ECCV 2022) 用于 raw 域去噪。

与仓库其他网络保持同样的接口：__init__(args)，forward(x: (b,C,H,W)) -> (b,C,H,W)。
输入输出均为 packed Bayer 4 通道，尺寸任意（建议能被 16 整除）。

关键组件：
  SimpleGate   —— 通道一分为二后相乘，等价非线性但无激活函数
  NAFBlock     —— LayerNorm + 1x1 → depthwise 3x3 → SimpleGate → 通道注意力 → 1x1，双残差
  LayerNorm2d  —— 对通道维做 LayerNorm（替代 BatchNorm，稳定且与 batch 无关）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=1, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, bias=True, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )
        self.conv4 = nn.Conv2d(c, c, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(c, c * 2, 1, 1, 0, bias=True)
        self.conv6 = nn.Conv2d(c, c, 1, 1, 0, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.conv5(x)
        x = self.sg(x)
        x = self.conv6(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.default_args(args)
        in_nc = self.args['in_nc']
        out_nc = self.args['out_nc']
        width = self.args['nf']
        enc_blk_nums = self.args['enc_blk_nums']
        dec_blk_nums = self.args['dec_blk_nums']
        middle_blk_num = self.args['middle_blk_num']

        self.intro = nn.Conv2d(in_nc, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, out_nc, 3, 1, 1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False),
                                          nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def default_args(self, args=None):
        self.args = {
            'in_nc': 4, 'out_nc': 4, 'nf': 32,
            'enc_blk_nums': [2, 2, 4, 8],
            'middle_blk_num': 12,
            'dec_blk_nums': [2, 2, 2, 2],
        }
        if args is not None:
            for k in args:
                self.args[k] = args[k]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def forward(self, inp):
        B, C, H, W = inp.shape
        x = self.check_image_size(inp)
        x = self.intro(x)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x[:, :, :H, :W] + inp   # 全局残差（NAFNet 标准做法）
        return x
