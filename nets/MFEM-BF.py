"""MPFEN 的 MFEM 备份实现，保留注意力特征热力图导出。"""

from functools import partial

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from nets.mobilenetv2 import mobilenetv2
from nets.xception import xception


class SEBlock(nn.Module):
    """通过全局平均池化和全连接层生成通道注意力权重。"""

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CrackAttentionModule(nn.Module):
    """依次应用通道、空间和边缘注意力，并保存首个样本的特征热力图。"""

    def __init__(self, channels):
        super(CrackAttentionModule, self).__init__()
        self.channel_attention = SEBlock(channels, reduction=8)

        # 空间注意力（对应 DGG）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # 边缘增强注意力（对应 BPU）
        self.edge_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 3, padding=1),
            nn.Sigmoid()
        )

        # 保存目录与序号（每次 forward 自增）
        self.save_dir = os.path.join("img", "MEFM")
        os.makedirs(self.save_dir, exist_ok=True)
        self._save_index = 0

    def _composite(self, fmaps):
        """对首个样本的通道绝对值取均值，归一化为 uint8 灰度图。"""
        with torch.no_grad():
            comp = fmaps.detach()
            comp = comp.abs().mean(dim=1, keepdim=True)
            comp = comp[0, 0]
            minv, maxv = float(comp.min()), float(comp.max())
            if maxv > minv:
                comp = (comp - minv) / (maxv - minv + 1e-6)
            else:
                comp = torch.zeros_like(comp)
            comp_np = (comp.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return comp_np

    def _save_heatmap(self, fmaps, name_prefix, idx_suffix):
        """将合成特征映射为 JET 彩色热力图并保存。"""
        comp_np = self._composite(fmaps)
        heatmap = cv2.applyColorMap(comp_np, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        save_path = os.path.join(self.save_dir, f"{name_prefix}{idx_suffix}.png")
        Image.fromarray(heatmap).save(save_path)

    def forward(self, x):
        """计算注意力特征，并以同一递增编号导出本次前向的热力图。"""
        idx_suffix = f"_{self._save_index:06d}"

        # TDE 输入
        self._save_heatmap(x, "TDE_input", idx_suffix)
        x_tde_out = self.channel_attention(x)
        # TDE 输出
        self._save_heatmap(x_tde_out, "TDE_output", idx_suffix)

        # DGG 输入
        self._save_heatmap(x_tde_out, "DGG_input", idx_suffix)
        spatial_weight = self.spatial_attention(x_tde_out)
        x_dgg_out = x_tde_out * spatial_weight

        # BPU 输入
        self._save_heatmap(x_dgg_out, "BPU_input", idx_suffix)
        edge_weight = self.edge_attention(x_dgg_out)
        x_bpu_out = x_dgg_out * edge_weight
        # BPU 输出
        self._save_heatmap(x_bpu_out, "BPU_output", idx_suffix)

        self._save_index += 1
        return x_bpu_out


class MobileNetV2(nn.Module):
    """按输出步幅配置空洞卷积的 MobileNetV2 特征提取器。"""

    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()

        # 在指定的 MobileNetV2 层中启用裂缝感知模块。
        model = mobilenetv2(pretrained, crack_aware_layers=[10, 11, 12, 13, 14, 15, 16])
        self.features = model.features[:-1]
        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        """取消指定卷积的下采样，并调整 3×3 卷积的膨胀率与填充。"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # 检查是否有stride属性
            if hasattr(m, 'stride') and m.stride == (2, 2):
                m.stride = (1, 1)
                if hasattr(m, 'kernel_size') and m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if hasattr(m, 'kernel_size') and m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        """提取并返回浅层特征与深层特征。"""
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class _DenseASPPBlock(nn.Module):
    """使用可分离空洞卷积生成新特征，并与输入沿通道维拼接。"""

    def __init__(self, in_channels, inter_channels, growth_rate, dilation_rate, drop_rate=0.1):
        super(_DenseASPPBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        )

        # 使用可分离卷积减少参数量
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, 3, padding=dilation_rate,
                      dilation=dilation_rate, groups=inter_channels, bias=False),
            nn.Conv2d(inter_channels, growth_rate, 1, bias=False)
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        if self.drop_rate > 0:
            output = F.dropout2d(output, p=self.drop_rate, training=self.training)
        return torch.cat([x, output], 1)


class CrackAwareDenseASPP(nn.Module):
    """密集堆叠多尺度空洞卷积，并通过裂缝注意力增强融合特征。"""

    def __init__(self, in_channels, out_channels, inter_channels=64, growth_rate=32,
                 dilation_series=[3, 6, 12, 18, 24]):
        super(CrackAwareDenseASPP, self).__init__()

        # 依次提取特征并加入裂缝感知处理。
        self.blocks = nn.ModuleList()
        channels = in_channels

        # 构建DenseASPP块
        for dilation in dilation_series:
            self.blocks.append(
                _DenseASPPBlock(channels, inter_channels, growth_rate, dilation_rate=dilation)
            )
            channels += growth_rate

        # 累计输入通道与各密集块新增的通道。
        final_channels = channels
        print(f"DenseASPP final channels: {final_channels}")  # 调试信息

        # 将密集拼接特征映射到目标通道数。
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(final_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 添加裂缝注意力
        self.crack_attention = CrackAttentionModule(out_channels)

    def forward(self, x):
        # 依次通过所有DenseASPP块
        """堆叠多尺度特征，经通道融合和裂缝注意力后输出。"""
        for block in self.blocks:
            x = block(x)

        # 融合特征
        output = self.fusion_conv(x)

        # 应用裂缝注意力
        output = self.crack_attention(output)

        return output


class EnhancedFeatureFusion(nn.Module):
    """对齐并融合高低层特征，使用边缘权重调制融合结果。"""

    def __init__(self, high_channels, low_channels, out_channels):
        super(EnhancedFeatureFusion, self).__init__()

        # 低层特征处理
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 高层特征处理
        self.high_conv = nn.Sequential(
            nn.Conv2d(high_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 多尺度融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(48 + 256, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # 边缘增强分支
        self.edge_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, 3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, high_feat, low_feat):
        # 处理低层特征
        """将高层特征上采样到低层尺寸，融合后应用边缘权重。"""
        low_feat = self.low_conv(low_feat)

        # 处理高层特征
        high_feat = self.high_conv(high_feat)

        # 上采样高层特征并融合
        high_feat = F.interpolate(high_feat, size=(low_feat.size(2), low_feat.size(3)),
                                  mode='bilinear', align_corners=True)

        # 特征融合
        fused_feat = torch.cat([high_feat, low_feat], dim=1)
        output = self.fusion_conv(fused_feat)

        # 边缘增强
        edge_weight = self.edge_branch(output)
        output = output * edge_weight

        return output


class DenseASPP(nn.Module):
    """密集连接多尺度空洞卷积，并将拼接特征映射到指定通道数。"""

    def __init__(self, in_channels, out_channels, inter_channels=64, growth_rate=32,
                 dilation_series=[3, 6, 12, 18, 24]):
        super(DenseASPP, self).__init__()
        self.blocks = nn.ModuleList()
        channels = in_channels
        for dilation in dilation_series:
            self.blocks.append(
                _DenseASPPBlock(channels, inter_channels, growth_rate, dilation_rate=dilation)
            )
            channels += growth_rate
        self.conv_out = nn.Sequential(
            nn.Conv2d(channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.conv_out(x)
        return x


class MPFEN(nn.Module):
    """MPFEN 分割网络，支持 MobileNetV2 和 Xception 骨干。"""

    def __init__(self, num_classes=1, backbone="mobilenet", pretrained=True,
                 downsample_factor=16, use_crack_aware=True):
        super(MPFEN, self).__init__()

        self.use_crack_aware = use_crack_aware

        if backbone == "xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        print(f"Backbone input channels: {in_channels}")  # 调试信息

        # 按配置选择裂缝感知或基础 DenseASPP。
        if use_crack_aware:
            self.aspp = CrackAwareDenseASPP(
                in_channels=in_channels,
                out_channels=256,
                inter_channels=64,
                growth_rate=32,
                dilation_series=[3, 6, 12, 18, 24]
            )
        else:
            # 使用基础 DenseASPP 提取多尺度特征。
            self.aspp = DenseASPP(
                in_channels=in_channels,
                out_channels=256,
                inter_channels=64,
                growth_rate=32,
                dilation_series=[3, 6, 12, 18, 24]
            )

        # 使用 SEBlock 调制通道响应。
        self.se_block = SEBlock(256, reduction=16)

        # 使用增强的特征融合模块
        if use_crack_aware:
            self.feature_fusion = EnhancedFeatureFusion(
                high_channels=256,
                low_channels=low_level_channels,
                out_channels=256
            )
        else:
            # 使用通道压缩、拼接与卷积完成基础特征融合。
            self.shortcut_conv = nn.Sequential(
                nn.Conv2d(low_level_channels, 48, 1),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True)
            )
            self.cat_conv = nn.Sequential(
                nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),

                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )

        # 最终分类层
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        """提取并融合多尺度特征，输出与输入空间尺寸一致的分割 logits。"""
        H, W = x.size(2), x.size(3)

        # 主干提取特征
        low_level_features, high_level_features = self.backbone(x)


        # 使用 DenseASPP 提取深层多尺度特征。
        aspp_features = self.aspp(high_level_features)

        # 应用 SEBlock 通道注意力。
        aspp_features = self.se_block(aspp_features)

        # 特征融合
        if self.use_crack_aware:
            # 使用增强的特征融合
            fused_features = self.feature_fusion(aspp_features, low_level_features)
        else:
            # 使用基础特征融合分支。
            low_level_features = self.shortcut_conv(low_level_features)
            aspp_features = F.interpolate(aspp_features,
                                          size=(low_level_features.size(2), low_level_features.size(3)),
                                          mode='bilinear', align_corners=True)
            fused_features = self.cat_conv(torch.cat((aspp_features, low_level_features), dim=1))

        # 最终分类
        output = self.cls_conv(fused_features)

        # 上采样到原始尺寸
        output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)

        return output


def create_crack_aware_mpfen(num_classes=1, backbone="mobilenet", pretrained=True, downsample_factor=16):
    """创建启用裂缝注意力和增强特征融合的 MPFEN。"""
    return MPFEN(num_classes=num_classes, backbone=backbone,
                   pretrained=pretrained, downsample_factor=downsample_factor,
                   use_crack_aware=True)


def create_standard_mpfen(num_classes=1, backbone="mobilenet", pretrained=True, downsample_factor=16):
    """创建使用基础 DenseASPP 和基础特征融合的 MPFEN。"""
    return MPFEN(num_classes=num_classes, backbone=backbone,
                   pretrained=pretrained, downsample_factor=downsample_factor,
                   use_crack_aware=False)
