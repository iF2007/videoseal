import sys
import time
import torch

# ==========================================
# ⚙️ 用户常用参数配置区 (User Configuration) ⚙️
# 可以在此处直接修改各项参数，用于测试不同环境
# ==========================================
class Config:
    # --- 环境测试配置 ---
    MODEL_NAME = "pixelseal"    # 要测试加载的模型名称: "videoseal", "pixelseal" 或 "chunkyseal"。默认"pixelseal"，因为水印鲁棒性和嵌入/提取性能最平衡
    FORCE_DEVICE = None         # 强制使用的设备，例如 "cpu", "cuda", "mps"；保持 None 则为自动检测最佳设备
# ==========================================


print("1. 正在加载基础库...")
import timm
import cv2
print(f"   [OK] torch version: {torch.__version__}")

print("\n2. 正在检测硬件加速器...")
if Config.FORCE_DEVICE:
    device = torch.device(Config.FORCE_DEVICE)
    print(f"   [OK] 用户强制按配置指定设备: {device}")
else:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   [OK] 当前自动检测分配设备: {device}")

print(f"\n3. 核心步骤：正在尝试加载 {Config.MODEL_NAME} 模型...")
print("   (注意：如果是第一次运行，此处会静默下载几百 MB 的权重，请观察网速)")
start_time = time.time()

try:
    import videoseal
    # 设置一个超时提醒逻辑（虽然 Python 没法真正异步中断加载，但能给提示）
    model = videoseal.load(Config.MODEL_NAME)
    model.to(device)
    print(f"   [OK] 模型加载成功！耗时: {time.time() - start_time:.2f} 秒")
except Exception as e:
    print(f"   [ERROR] 加载失败: {e}")

print("\n4. 正在检查其他关键依赖...")
try:
    import decord
    print("   [OK] decord (视频读取) 已就绪")
    from omegaconf import OmegaConf
    print("   [OK] omegaconf (配置管理) 已就绪")
except ImportError as e:
    print(f"   [MISSING] 缺少模块: {e}")

print("\n--- 诊断结束 ---")
