"""评估 MPFEN 分割指标，并将裂缝区域以红色叠加到原图。

预测 PNG 保存类别索引；二分类时像素值为 0 或 1，因此直接查看接近全黑。
评估样本由 Test0717/ImageSets/Segmentation/val.txt 指定。
"""

import os

# 使用 Qt 离屏模式，支持无显示器环境中的结果生成。
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PIL import Image
from tqdm import tqdm

from mpfen import MPFEN
from utils.utils_metrics import compute_mIoU, show_results

import numpy as np

if __name__ == "__main__":
    # 0：生成预测并评估；1：仅生成预测；2：仅评估已有预测。
    miou_mode = 0

    # 类别总数包含背景；名称顺序必须与标签像素值一致。
    num_classes = 2

    name_classes = ["_background_", "crack"]

    # 数据集根目录及验证集图像、标签路径。
    VOCdevkit_path = 'VOCdevkit'

    image_ids = open(os.path.join(VOCdevkit_path, "Test0717/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "Test0717/SegmentationClass/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    # 保存类别索引 PNG，用于后续指标计算。
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        mpfen = MPFEN(model_path="logs/best_epoch_weights.pth")
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "Test0717/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = mpfen.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    # 根据预测与真值计算混淆矩阵及各类别指标。
    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir, pred_dir, image_ids, num_classes, name_classes
        )
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

    # 将已有预测中的裂缝区域叠加到原图；类别索引 1 表示裂缝。
    idx_crack = 1
    # 红色叠加权重，取值范围为 [0, 1]。
    alpha = 0.5
    image_dir = os.path.join(VOCdevkit_path, "Test0717/JPEGImages")
    vis_on_orig_dir = os.path.join(miou_out_path, 'colored-results-on-image')
    os.makedirs(vis_on_orig_dir, exist_ok=True)

    print("生成裂缝区域亮红色叠加图中...")
    for image_id in tqdm(image_ids):
        mask_path = os.path.join(pred_dir, image_id + ".png")
        orig_img_path = os.path.join(image_dir, image_id + ".jpg")

        if not os.path.exists(mask_path) or not os.path.exists(orig_img_path):
            print(f"Warning: {image_id} 缺失掩码或原图，已跳过")
            continue

        mask = np.array(Image.open(mask_path))
        orig_img = np.array(Image.open(orig_img_path).convert("RGB"))

        # 使用最近邻插值匹配原图尺寸，保持类别索引不变。
        if mask.shape[:2] != orig_img.shape[:2]:
            mask = np.array(Image.open(mask_path).resize(
                (orig_img.shape[1], orig_img.shape[0]), Image.NEAREST))

        # 仅为裂缝区域生成红色覆盖层。
        red_mask = np.zeros_like(orig_img)
        red_mask[mask == idx_crack] = [255, 0, 0]

        # 在裂缝区域混合原图与红色覆盖层。
        vis = orig_img.copy()
        crack_pixels = (mask == idx_crack)
        vis[crack_pixels] = ((1 - alpha) * vis[crack_pixels] + alpha * red_mask[crack_pixels]).astype(np.uint8)

        Image.fromarray(vis).save(os.path.join(vis_on_orig_dir, image_id + ".jpg"))

    print(f"裂缝区域亮红色叠加图已生成，保存至 {vis_on_orig_dir}")
