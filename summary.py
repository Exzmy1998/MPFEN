"""打印 MPFEN 网络结构，并统计参数量与浮点运算量。"""

import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.mpfen import MPFEN

if __name__ == "__main__":
    # 输入分辨率、类别总数及主干网络配置。
    input_shape = [512, 512]
    num_classes = 2
    backbone = 'mobilenet'

    # 优先使用 CUDA，未提供 CUDA 时使用 CPU。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MPFEN(num_classes=num_classes, backbone=backbone, downsample_factor=16, pretrained=False).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))

    # 以单张随机输入统计计算量和参数量。
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(model.to(device), (dummy_input, ), verbose=False)

    # 按一次乘法与一次加法各计一次运算的口径，将 THOP 统计量乘以 2。
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
