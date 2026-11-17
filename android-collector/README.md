# DarkFrameCollector

为 PNNP（[Learning Physics-Informed Noise Models from Dark Frames](https://arxiv.org/abs/2310.09126)）采集训练数据的 Android 应用：**GT 采集（干净图像）** + **暗帧采集（噪声标定）** 双入口。

## 主界面

| 入口 | 用途 | 输出 |
|---|---|---|
| **GT 采集** | 去噪网络学习图像内容 | `gt/<模式>/<场景>/*.dng` |
| **暗帧采集** | PNNP 学习噪声模型 | `bias/ISO{iso}/*.dng` |
| **平场采集** | PTC 标定 shot noise 增益 K | `flat/ISO{iso}/*.dng` |

## 平场采集（PTC 标定）

为拟合 `Var_DN = g·μ_DN + σ_read²`（K = 1/g）采集数据：

1. **对准均匀表面**（白墙/匀光板，无纹理无阴影），光照全程恒定
2. 两种曝光网格：
   - **自动倍增**（推荐）：从起始曝光逐档 ×2，帧均值达 0.7×白电平自动停止——不用预估结束曝光
   - **手动网格**：起始/结束曝光 + 步数，log 均匀分布
3. 每档 ≥10 帧（PTC 统计需求），逐 ISO 循环
4. 质检：平场均匀性（std/mean > 15% 警告）+ 过曝警告
5. 离线拟合：逐档算时间方差与均值，线性拟合得 g，截距即 σ_read²（可与暗帧实测交叉验证）；多 ISO 拟合 K = K_k·ISO − K_b

## GT 采集（两种模式）

低光条件由训练时 PNNP 代理**在线合成**，GT 只需干净——不需要暗环境、不需要长曝光。

1. **暗室多帧平均**：暗环境下对静态场景固定参数连拍 N 帧，离线对齐平均得高 SNR GT（SNR 提升 √N）。预设 ISO800 / 25ms / 25 帧。适用于光照受限但场景可控的情况。
2. **亮环境低 ISO 直拍**：明亮环境低 ISO 短曝光直接拍。预设 ISO50 / 10ms / 3 帧。最快，优先用这个。

每次"开始采集"= 一个场景一个目录（填场景名）。质检：过曝（均值 >85% 白电平）与亮环境欠曝警告。

**离线平均流程**（burst_avg 模式，PC 端）：暗帧校正（减 BLE/dark shading）→ 对齐（ECC/单应）→ 平均。对齐前务必先做暗帧校正，否则 FPN 会烙进 GT。

## 暗帧采集

盖镜头，ISO × 曝光网格逐档连拍。规范：
- 遮光从严（镜头盖最佳，布料可能透红外）；温度稳定后再采；每档 ≥5 帧
- 质检：raw 均值过高（漏光）自动警告
- 输出结构与 PNNP 训练代码 `data_process/dark_datasets.py` 的 `bias_dir/<ISO>/*` 约定一致，rawpy 直接读取

## 使用

1. Android Studio 打开 `android-collector/`，构建安装（Android 8.0+、支持 Camera2 RAW + 手动控制）
2. 采集完成后从
   `Android/data/com.example.darkframecollector/files/Pictures/DarkFrameCollector/<时间戳>/`
   取出目录（暗帧的 `bias/` 与 GT 的 `gt/` 分属不同时间戳目录，可合并使用）

## 数据接入训练

- 暗帧 → `trainer_PNNP_proxy.py`（改 runfile 的 `bias_dir`）
- GT → 去噪训练 runfile 的数据配置（如 `Proxy_Dataset` 指向 GT 目录）
- 每批数据附 `session.json`：设备型号、ISO/曝光范围、白/黑电平、逐帧元数据

## 实现说明

纯 Camera2 API，无第三方依赖，minSdk 23。三个 Activity 共用 `BaseCaptureActivity` 的相机/落盘管线（burst 连拍、按 `SENSOR_TIMESTAMP` 双向配对结果与图像、`DngCreator` 存 DNG）；`MainActivity` 为入口，`DarkActivity` / `GtActivity` 各自实现 UI 与质检逻辑。
