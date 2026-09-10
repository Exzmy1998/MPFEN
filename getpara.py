"""统计 MPFEN 参数量、计算量及预热后的平均推理耗时。"""

import torch
import time
from ptflops import get_model_complexity_info

from nets.mpfen import MPFEN

# 测试配置：优先使用 CUDA，输入维度依次为通道、高度和宽度。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_RES = (3, 512, 512)
BATCH_SIZE = 1
# 正式计时的推理次数。
NUM_RUNS = 50

# 使用未加载预训练权重的模型，并切换到评估模式。
model = MPFEN(
    num_classes=2,
    backbone="mobilenet",
    pretrained=False,
    downsample_factor=16
).to(DEVICE)
model.eval()

# 统计参与梯度计算的参数数量。
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params:,}  ({total_params/1e6:.2f} M)")


def input_constructor(input_res):
    """构造与模型位于同一设备的批量随机输入。"""
    return torch.randn(BATCH_SIZE, *input_res).to(DEVICE)


# 通过 ptflops 统计并输出计算量和参数量。
flops_str, params_str_ptflops = get_model_complexity_info(
    model,
    INPUT_RES,
    as_strings=True,
    print_per_layer_stat=False,
    input_constructor=input_constructor
)

print(f"FLOPs: {flops_str}")
print(f"Params(由ptflops统计): {params_str_ptflops}")

# 使用固定批次的随机输入测量推理时间。
x = torch.randn(BATCH_SIZE, *INPUT_RES).to(DEVICE)

with torch.no_grad():
    # 先预热 10 次，减少首次初始化对计时的影响。
    for _ in range(10):
        _ = model(x)

    # CUDA 操作为异步执行，计时边界需要同步。
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    for _ in range(NUM_RUNS):
        _ = model(x)

    # CUDA 操作为异步执行，计时边界需要同步。
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

# 平均耗时按批次计算；默认批次大小为 1。
avg_time_ms = (end - start) / NUM_RUNS * 1000
fps = 1000 / avg_time_ms

print(f"推理时间: {avg_time_ms:.2f} ms/张")
print(f"FPS: {fps:.2f}")
