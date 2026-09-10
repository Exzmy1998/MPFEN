"""MPFEN 推理接口，提供图像分割、类别统计和模型导出。"""

import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.aco_optimizer import AntColonyOptimizer
from nets.mpfen import MPFEN as MPFENModel
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


class MPFEN(object):
    """封装 MPFEN 的权重加载、图像分割、速度评估和 ONNX 导出。"""

    _defaults = {
        # 权重路径及网络配置应与训练时一致。
        "model_path": 'logs/best_epoch_weights.pth',
        "num_classes": 2,  # 包含背景类；二分类对应背景与裂缝。
        "backbone": "mobilenet",  # 可选 mobilenet 或 xception。
        "input_shape": [512, 512],  # [高度, 宽度]。
        "downsample_factor": 16,  # 可选 8 或 16。
        # 可视化模式：0 为叠加原图，1 为类别色彩图，2 为去除背景。
        "mix_type": 0,
        "cuda": True,  # 无可用 GPU 时设为 False。
        "use_aco": False,  # 是否在 detect_image 中启用二分类 ACO 后处理。
    }

    def __init__(self, **kwargs):
        """初始化推理配置并加载模型；kwargs 可覆盖默认参数。"""
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # 为分割类别分配可视化颜色。
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # 构建网络并载入权重。
        self.generate()

        if self.use_aco:
            print("ACO post-processing is enabled.")
            self.aco_optimizer = AntColonyOptimizer(
                num_ants=2000,
                max_iter=50,
                beta=3.0,
                rho=0.1,
                start_threshold=0.8
            )

        show_config(**self._defaults)

    def generate(self, onnx=False):
        """构建 MPFEN 网络并加载权重，导出时保留未包装的网络。"""
        # 按当前配置构建网络并加载完整模型参数。
        self.net = MPFENModel(num_classes=self.num_classes, backbone=self.backbone, downsample_factor=self.downsample_factor, pretrained=False)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, count=False, name_classes=None):
        """返回可视化分割图；count=True 时按 name_classes 输出像素统计。"""
        # 将输入统一转换为 RGB，兼容灰度图等图像模式。
        image       = cvtColor(image)
        # 保留原图，用于叠加分割结果。
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        # 等比例缩放并填充至网络输入尺寸。
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        # 归一化并转换为 [1, C, H, W] 张量布局。
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # 执行前向推理。
            pr = self.net(images)[0]
            # 计算逐像素类别概率。
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            # 裁去缩放时添加的填充区域。
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            # 将类别概率图恢复至原图尺寸。
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            # 准备后处理使用的类别概率图。
            pr_prob = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

            if self.use_aco:
                # ACO 后处理仅支持背景与目标二分类。
                if self.num_classes != 2:
                    print("[Warning]ACO post - processing is designed  for binary segmentation(num_classes=2).")
                    print("Falling back to standard argmax prediction.")
                    pr = pr_prob.argmax(axis=-1)
                else:
                    print("Starting ACO post - processing...")
                    # 提取类别 1（裂缝）的概率图。
                    target_probability_map = pr_prob[:, :, 1]

                    # 运行蚁群优化，得到信息素图。
                    optimized_map = self.aco_optimizer.run(target_probability_map)

                    # 信息素强度高于原始背景概率的像素判定为目标。
                    pr = (optimized_map > pr_prob[:, :, 0]).astype(np.uint8)
                    print("ACO post-processing finished.")
            else:
                pr = pr.argmax(axis=-1)

        # 统计各类别像素数量及占比。
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            # 转换为 PIL 图像。
            image   = Image.fromarray(np.uint8(seg_img))
            # 将分割色彩图与原图叠加。
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            # 转换为 PIL 图像。
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            # 转换为 PIL 图像。
            image = Image.fromarray(np.uint8(seg_img))

        return image

    def get_FPS(self, image, test_interval):
        """预热后重复推理，返回每次推理的平均耗时，单位为秒。"""
        # 将输入统一转换为 RGB，兼容灰度图等图像模式。
        image       = cvtColor(image)
        # 等比例缩放并填充至网络输入尺寸。
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        # 归一化并转换为 [1, C, H, W] 张量布局。
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # 执行前向推理。
            pr = self.net(images)[0]
            # 计算逐像素类别概率并取最大概率类别。
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            # 裁去缩放时添加的填充区域。
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # 执行前向推理。
                pr = self.net(images)[0]
                # 计算逐像素类别概率并取最大概率类别。
                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
                # 裁去缩放时添加的填充区域。
                pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        """导出并检查 ONNX 模型，可选是否简化计算图。"""
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # 输入布局为 [B, C, H, W]。
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]

        # 导出固定输入尺寸的 ONNX 模型。
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # 检查导出模型的结构合法性。
        model_onnx = onnx.load(model_path)
        onnx.checker.check_model(model_onnx)

        # 按需简化 ONNX 计算图。
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_miou_png(self, image):
        """生成用于 mIoU 评估的单通道类别索引图，不应用 ACO 后处理。"""
        # 将输入统一转换为 RGB，兼容灰度图等图像模式。
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        # 等比例缩放并填充至网络输入尺寸。
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        # 归一化并转换为 [1, C, H, W] 张量布局。
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # 执行前向推理。
            pr = self.net(images)[0]
            # 计算逐像素类别概率。
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            # 裁去缩放时添加的填充区域。
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            # 将类别概率图恢复至原图尺寸。
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            # 选择每个像素概率最大的类别。
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image
