# VideoSeal Watermarking Test Suite

欢迎使用 VideoSeal 专业级隐藏水印测试套件。本目录包含了一套基于深度学习和 BCH 纠错码算法的端到端图像/视频盲水印测试工具流程。

该测试套件实现了**高鲁棒性**的盲水印注入和提取，并具备完善的抗破坏评估模块。所有的脚本**均已解耦了命令行传参**，用户直接在代码内指定的 `USER CONFIGURATION` 区域修改参数即可执行。

---

## 🛠 文件结构及执行流

工作流通常按照以下顺序执行：

### 1. `init_env.py` (环境探针与基础配置诊断)
**作用**：当你刚配置好 Conda 环境或在新的机器上使用 VideoSeal 库时，用于诊断深度学习依赖、硬件加速器（CUDA/MPS）的就绪状态。它会自动尝试下载并加载你指定体积的检测模型权重。
**使用方法**：
修改代码顶部的 `Config` 类，确定你要测试下载的模型名称（如 `pixelseal`），然后运行：
```bash
python init_env.py
```

### 2. `bch_config_generator.py` (BCH 纠错空间发生器)
**作用**：水印系统物理底层能写入的信息量是固定的（256 bits），本脚本根据你的实际业务要求，自动帮你计算及分配`信息净载荷空间`和`纠错码(ECC)空间`的黄金比例。**最大水印长度越短，其安全性和容错率呈指数级提升。**
**使用方法**：
- 编辑代码顶部的 `UserConfig`，设置你的 `STRENGTH`（水印强度） 和预期的 `MAX_WATERMARK_LENGTH`（写入字符串字符上限长度）。
- 运行：
```bash
python bch_config_generator.py
```
*执行后会在当前目录下生成一份 `bch_config.json` 架构文件供后续的脚本调用。*

### 3. `demo.py` (水印注入/提取核心引擎)
**作用**：基于生成的 `bch_config.json`，自动读取指定的图像，在频域/像素域进行不可见的神经网络注入，并将其以高质量 JPEG 落盘。它能够直接检验刚打入的水印能否在不破坏的情况下无损读出。
**使用方法**：
- 在代码块底部 `USER CONFIGURATION` 区，配置 `INPUT_PATHStr`（待测试图像或目录，如本项目的 `val2017_subset`）和 `WATERMARK_TEXT`。
- 运行：
```bash
python demo.py
```
*水影处理后的图像默认落盘在其父目录名为带 `_wm` 的新建文件夹中，如：`val2017_subset_wm`。*

### 4. `test_attacks.py` (抗破坏对抗鲁棒性验收测试)
**作用**：对已打好水印的图片施加各类极端的破坏性前处理操作（包括但不限于 `单点攻击` 及 `组合攻击`），以此在恶劣环境下榨取水印的存活能力。它会自动联动 `demo.py` 配置好的底层文本。
**内置的典型攻击组合包括：**
- 高效 JPEG 重压缩 (Quality 50 / 30)
- 高斯随机底噪与模糊 (Gaussian_Noise / Gaussian_Blur)
- 画幅几何切割 (Center_Crop 50% / 80%)
- 仿射变换与形变 (Scale / Rotation)
- 过饱和度与明暗变异亮度调整
- 全面的 `多重恶劣条件组合`
**使用方法**：
- 在脚本底部的 `USER CONFIGURATION` 区修改你要读取并测试的带水印残差测试图片（如：`val2017_subset_wm/000000001675_wm.jpg`）。
- 运行：
```bash
python test_attacks.py
```
*脚本会自动输出一份全维度的鲁棒性得分摘要及 BCH 的恢复挽救位数量（ECC recovered bits）报告。*

---

## 💡 最佳实践与调优指南

如果在使用中，你发现对抗攻击测试的 `FAIL` 较多，建议参照以下逻辑进行策略修改：
1. 打开 `bch_config_generator.py`，尽可能**调低** `MAX_WATERMARK_LENGTH`（比如 10-16 个字符之间），把挤出的 20 多 Bytes 全额交给 BCH 纠错算法使用。
2. 在 `bch_config_generator.py` 中，适度**调高** `STRENGTH`。经验值建议处于 `1.5 - 2.5` 之间，它会使注入的高频扰动潜行更深，抵抗大参数的几何改变拦截。
3. **完成配置**修改后，请务必执行三连重新打下地基：
   `python bch_config_generator.py` (重构参数) -> `python demo.py` (重刻图片) -> `python test_attacks.py` (重启测试)

---
*Environment Required: Conda - `videoseal`*
