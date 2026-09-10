"""MPFEN 图像预处理、随机种子与预训练权重工具。"""

import random

import numpy as np
import torch
from PIL import Image


def cvtColor(image):
    """将输入图像转换为 RGB 格式。"""
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size):
    """等比例缩放并居中填充，返回图像及有效区域的宽、高。"""
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh


def get_lr(optimizer):
    """返回优化器首个参数组的学习率。"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_everything(seed=11):
    """设置 Python、NumPy 与 PyTorch 的随机种子及 cuDNN 配置。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, rank, seed):
    """使用进程编号与基础种子初始化数据加载进程的随机状态。"""
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def preprocess_input(image):
    """将像素值原地缩放至 [0, 1]。"""
    image /= 255.0
    return image


def show_config(**kwargs):
    """以表格形式打印配置项。"""
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def download_weights(backbone, model_dir="./model_data"):
    """从上游发布地址下载指定骨干网络的预训练权重。"""
    import os
    from torch.hub import load_state_dict_from_url

    # 保留骨干网络权重的真实上游地址，项目命名不影响下载来源。
    download_urls = {
        'mobilenet' : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'xception'  : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
    }
    url = download_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)
