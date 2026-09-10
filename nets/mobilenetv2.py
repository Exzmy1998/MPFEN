"""支持裂缝感知倒残差块的 MobileNetV2 骨干网络。"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

BatchNorm2d = nn.BatchNorm2d


def conv_bn(inp, oup, stride):
    """构建 3×3 卷积、批归一化与 ReLU6 激活层。"""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    """构建用于通道映射的 1×1 卷积、批归一化与 ReLU6 激活层。"""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class EdgeAwareModule(nn.Module):
    """融合输入特征与双向边缘响应，并归一化拼接结果。"""

    def __init__(self, channels):
        super(EdgeAwareModule, self).__init__()
        self.edge_conv_x = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.edge_conv_y = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels * 2)

        # 初始化为类似Sobel算子的边缘检测核
        self._initialize_sobel_weights()

    def _initialize_sobel_weights(self):
        """以 x、y 方向的 Sobel 核初始化边缘卷积权重。"""
        with torch.no_grad():
            # 沿 x 方向计算梯度。
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            # 沿 y 方向计算梯度。
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

            # 将 3x3 核扩展为 (out_channels, in_channels, 3, 3)
            sobel_x = sobel_x.view(1, 1, 3, 3).repeat(self.edge_conv_x.out_channels, self.edge_conv_x.in_channels, 1, 1)
            sobel_y = sobel_y.view(1, 1, 3, 3).repeat(self.edge_conv_y.out_channels, self.edge_conv_y.in_channels, 1, 1)

            # 整体复制权重
            self.edge_conv_x.weight.copy_(sobel_x)
            self.edge_conv_y.weight.copy_(sobel_y)

    def forward(self, x):
        edge_x = self.edge_conv_x(x)
        edge_y = self.edge_conv_y(x)
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        combined = torch.cat([x, edge_magnitude], dim=1)
        return self.bn(combined)


class DirectionalConv(nn.Module):
    """融合水平、垂直和两个可学习方形卷积分支的特征。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(DirectionalConv, self).__init__()
        quarter_channels = max(1, out_channels // 4)

        # 不同方向的卷积核
        self.conv_horizontal = nn.Conv2d(in_channels, quarter_channels, (1, 7), stride=stride, padding=(0, 3),
                                         bias=False)
        self.conv_vertical = nn.Conv2d(in_channels, quarter_channels, (7, 1), stride=stride, padding=(3, 0), bias=False)
        self.conv_diagonal1 = nn.Conv2d(in_channels, quarter_channels, 3, stride=stride, padding=1, bias=False)
        self.conv_diagonal2 = nn.Conv2d(in_channels, out_channels - 3 * quarter_channels, 3, stride=stride, padding=1,
                                        bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

        # 提供输出步幅调整所需的卷积属性。
        self.stride = (stride, stride)
        self.kernel_size = (3, 3)

    def forward(self, x):
        h_feat = self.conv_horizontal(x)
        v_feat = self.conv_vertical(x)
        d1_feat = self.conv_diagonal1(x)
        d2_feat = self.conv_diagonal2(x)

        combined = torch.cat([h_feat, v_feat, d1_feat, d2_feat], dim=1)
        return self.relu(self.bn(combined))


class TextureAwareModule(nn.Module):
    """融合纹理卷积与局部对比度卷积的响应。"""

    def __init__(self, channels, stride=1):
        super(TextureAwareModule, self).__init__()
        self.texture_conv = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.contrast_conv = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU6(inplace=True)

        # 提供输出步幅调整所需的卷积属性。
        self.stride = (stride, stride)
        self.kernel_size = (3, 3)

    def forward(self, x):
        # 通过局部均值估计对比度；过小的特征图使用自身作为均值。
        if x.size(2) > 3 and x.size(3) > 3:
            mean_pool = F.avg_pool2d(x, 3, stride=1, padding=1)
        else:
            mean_pool = x
        contrast = torch.abs(x - mean_pool)

        texture_feat = self.texture_conv(x)
        contrast_feat = self.contrast_conv(contrast)

        combined = texture_feat + contrast_feat
        return self.relu(self.bn(combined))


class CrackAwareInvertedResidual(nn.Module):
    """在倒残差结构中按配置加入边缘感知特征。"""

    def __init__(self, inp, oup, stride, expand_ratio, use_crack_aware=True):
        super(CrackAwareInvertedResidual, self).__init__()
        self.stride = stride
        self.use_crack_aware = use_crack_aware
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if not use_crack_aware:
            # 构建标准倒残差分支。
            if expand_ratio == 1:
                self.conv = nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    BatchNorm2d(oup),
                )
        else:
            # 构建包含边缘感知的倒残差分支。
            # 依次提取特征并加入裂缝感知处理。
            if expand_ratio == 1:
                # 扩展倍率为 1 时跳过逐点通道扩展。
                self.expand_conv = None
                expand_channels = inp
            else:
                self.expand_conv = nn.Sequential(
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True)
                )
                expand_channels = hidden_dim

            # 深度卷积 + 裂缝感知
            self.dwconv = nn.Conv2d(expand_channels, expand_channels, 3, stride, 1, groups=expand_channels, bias=False)
            self.bn1 = BatchNorm2d(expand_channels)

            # 裂缝感知模块
            self.edge_aware = EdgeAwareModule(expand_channels)

            # 融合后的通道数是原来的2倍
            fused_channels = expand_channels * 2

            # 降维到输出通道
            self.project_conv = nn.Sequential(
                nn.Conv2d(fused_channels, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup)
            )

    def forward(self, x):
        if not self.use_crack_aware:
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)
        else:
            identity = x

            # 扩展
            if self.expand_conv is not None:
                x = self.expand_conv(x)

            # 深度卷积
            x = F.relu6(self.bn1(self.dwconv(x)), inplace=True)

            # 边缘感知
            x = self.edge_aware(x)

            # 投影到输出维度
            x = self.project_conv(x)

            # 残差连接
            if self.use_res_connect:
                return identity + x
            else:
                return x
class InvertedResidual(nn.Module):
    """由通道扩展、逐通道卷积和线性投影组成的倒残差块。"""

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """支持裂缝感知配置的 MobileNetV2 分类骨干。"""

    def __init__(self, n_class=1000, input_size=224, width_mult=1., crack_aware_layers=None):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        crack_block = CrackAwareInvertedResidual
        input_channel = 32
        last_channel = 1280

        # 指定启用裂缝感知的层编号；默认选择中后段层。
        if crack_aware_layers is None:
            crack_aware_layers = [10, 11, 12, 13, 14, 15, 16]

        interverted_residual_setting = [
            # t：扩展倍率；c：输出通道数；n：重复次数；s：首块步幅。
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        # 第一层：普通卷积
        self.features = [conv_bn(3, input_channel, 2)]

        layer_idx = 1
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                layer_idx += 1
                if i == 0:
                    if layer_idx in crack_aware_layers:
                        self.features.append(
                            crack_block(input_channel, output_channel, s, expand_ratio=t, use_crack_aware=True))
                    else:
                        self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    if layer_idx in crack_aware_layers:
                        self.features.append(
                            crack_block(input_channel, output_channel, 1, expand_ratio=t, use_crack_aware=True))
                    else:
                        self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        """提取特征并全局平均池化，输出分类 logits。"""
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """按层类型初始化卷积、批归一化和分类器参数。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def load_url(url, model_dir='./model_data', map_location=None):
    """优先读取本地缓存，否则从上游地址下载预训练权重。"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url, model_dir=model_dir)


def mobilenetv2(pretrained=False, **kwargs):
    """创建 MobileNetV2，并按需加载可匹配的预训练权重。"""
    model = MobileNetV2(n_class=1000, **kwargs)
    if pretrained:
        try:
            # 从上游权重仓库加载参数；下载地址保留真实仓库名称。
            pretrained_dict = load_url(
                'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar')
            model_dict = model.state_dict()

            # 过滤掉不匹配的键
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"成功加载 {len(pretrained_dict)} 个预训练参数")
        except Exception as e:
            print(f"预训练权重加载失败: {e}")
            print("使用随机初始化权重")

    return model
