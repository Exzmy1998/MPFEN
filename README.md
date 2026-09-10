# MPFEN

基于 PyTorch 的裂缝语义分割项目，提供模型训练、图像与视频预测、分块预测、mIoU 评估和 ONNX 导出。

当前主网络结合 MobileNetV2 或 Xception 主干、DenseASPP、注意力模块与高低层特征融合。`use_crack_aware` 控制是否启用裂缝感知的多尺度特征提取与增强融合分支。

## 项目结构

```text
MPFEN/
├── nets/
│   ├── mpfen.py                # 主网络、注意力模块与特征融合
│   ├── mpfen_training.py       # 损失函数、初始化与学习率调度
│   ├── mpfen_bf.py             # 备用网络实现
│   ├── mpfen_dense_bf.py       # 备用网络实现
│   ├── mobilenetv2.py          # MobileNetV2 主干
│   ├── xception.py             # Xception 主干
│   └── aco_optimizer.py        # 可选蚁群后处理
├── utils/                     # 数据加载、训练回调、指标与通用工具
├── model_data/                # 预训练权重与导出模型
├── mpfen.py                   # 推理接口
├── train.py                   # 训练入口
├── predict.py                 # 单图分块预测及其他预测模式
├── predict2.py                # 支持目录分块预测的入口
├── get_miou.py                # 验证集指标评估
├── zmy-get_miou.py            # 指标评估与叠加图输出
├── summary.py                 # 网络结构与计算量统计
├── getpara.py                 # 参数量、计算量与耗时统计
├── json_to_dataset.py         # Labelme 标注转换
└── voc_annotation.py          # 数据集划分与标签检查
```

## 环境与权重

核心依赖包括 PyTorch、torchvision、NumPy、Pillow、OpenCV、SciPy、Matplotlib、tqdm、TensorBoard 和 albumentations。`requirements.txt` 保留原项目的依赖声明，使用时还需核对上述依赖；`requirments2.txt` 是历史环境导出，包含本机路径及其他项目依赖，不宜直接用于新环境。

附加功能所需依赖：参数与计算量统计使用 `thop`、`torchsummary`、`ptflops`，标注转换使用 `labelme`，ONNX 导出使用 `onnx`，模型简化另需 `onnxsim`。

训练与推理默认读取 `logs/best_epoch_weights.pth`，运行前请准备与网络结构匹配的权重，或修改 `model_path`。`model_data/mpfen_mobilenetv2.pth` 是统一命名后保留的历史预训练权重，文件内容未改变；它不代表当前增强网络训练完成的权重，不能仅凭文件名判断是否可直接用于推理。训练脚本按参数名和形状加载匹配项，推理接口要求完整匹配。

从主干预训练权重开始训练时，设置 `model_path = ''`、`pretrained = True`；随机初始化训练时，设置 `model_path = ''`、`pretrained = False`、`Freeze_Train = False`。

## 数据准备

当前脚本使用以下目录结构，数据子目录名为 `Test0717`：

```text
VOCdevkit/
└── Test0717/
    ├── JPEGImages/            # 原始图像，后缀为 .jpg
    ├── SegmentationClass/     # 类别索引标签，后缀为 .png
    └── ImageSets/
        └── Segmentation/
            ├── train.txt     # 每行一个样本名，不含扩展名
            ├── val.txt
            ├── trainval.txt
            └── test.txt
```

图像与标签的文件主名必须一致。二分类标签中，背景像素为 `0`，裂缝像素为 `1`；`num_classes = 2` 包含背景类。用于展示的 0/255 二值图不能直接作为该格式的训练标签。

按需配置并运行 `json_to_dataset.py` 转换 Labelme 标注，该脚本的输入输出目录需与上述数据目录对齐。放好图像与标签后，运行 `python voc_annotation.py` 生成划分文件。

如采用其他数据子目录名，需要同步修改训练、评估、数据加载及回调中的目录配置。

## 训练

在 `train.py` 中设置模型、权重、数据路径、输入尺寸、训练轮数和批次大小，然后运行：

```bash
python train.py
```

- `num_classes`：含背景在内的类别总数。
- `backbone`：`mobilenet` 或 `xception`。
- `downsample_factor`：`8` 或 `16`，推理时应与训练一致。
- `Freeze_Train`：是否先冻结主干，再解冻训练。
- `UnFreeze_Epoch`：训练结束轮次。
- `Cuda`：是否使用 GPU，CPU 运行时设为 `False`。

当前训练脚本的 `downsample_factor` 默认值为 `8`，推理接口默认值为 `16`，使用权重前需按其训练配置统一设置。权重保存在 `logs/`，日志保存在对应时间戳的子目录中。当前续训方式只加载模型参数，不恢复优化器和梯度缩放器状态。

## 预测与导出

在 `mpfen.py` 的 `_defaults` 中配置权重及网络参数，或在创建推理实例时传入参数：

```python
from PIL import Image
from mpfen import MPFEN

model = MPFEN(
    model_path="logs/best_epoch_weights.pth",
    num_classes=2,
    backbone="mobilenet",
    downsample_factor=8,
    cuda=False,
)
result = model.detect_image(Image.open("img/example.jpg"))
result.save("img/result.png")
```

`mix_type = 0` 叠加原图，`1` 输出类别色彩图，`2` 保留目标区域原图。`use_aco=True` 会在 `detect_image` 中启用二分类蚁群后处理；评估接口 `get_miou_png` 输出网络直接预测的类别索引图。

| mode | 功能 |
| --- | --- |
| `predict` | 交互输入单张图片路径，分块预测并拼接结果 |
| `video` | 视频或摄像头预测 |
| `fps` | 计算平均推理耗时与 FPS |
| `dir_predict` | 目录批量预测；`predict2.py` 在此模式下分块拼接 |
| `export_onnx` | 导出 ONNX 模型，可配置是否简化计算图 |

配置所需模式及输入输出路径后，运行 `python predict.py` 或 `python predict2.py`。

## 评估与模型统计

在 `get_miou.py` 中核对 `num_classes`、`name_classes` 和数据路径，确认推理配置与权重一致，然后运行 `python get_miou.py`。

`miou_mode = 0` 表示预测并计算指标，`1` 表示仅生成预测标签，`2` 表示使用已有预测标签计算指标。结果输出至 `miou_out/`。`zmy-get_miou.py` 还会生成原图与预测结果的叠加图。

`summary.py` 和 `getpara.py` 用于网络结构、参数量或计算量统计，运行前需核对脚本中的模型与设备配置。

## 来源与许可

项目保留 [MIT 许可证](LICENSE) 中的原作者署名。代码中的主干权重下载地址保留真实资源路径：[上游预训练权重发布页](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/tag/v1.0)。这些外部地址中的名称属于资源来源，不是本项目模型接口。MPFEN 的准确率以本项目实际训练与评估结果为准。
