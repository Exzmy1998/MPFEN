"""将 LabelMe JSON 标注转换为 VOC 格式图像与类别索引标签。

标签 PNG 使用调色板显示颜色，像素值实际表示类别索引。
当前转换接口沿用 LabelMe 3.16.7，类别顺序由 classes 配置决定。
"""

import base64
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils

if __name__ == '__main__':
    # 配置输出目录及统一类别顺序；背景类别固定为索引 0。
    jpgs_path = "datasets/JPEGImages"
    pngs_path = "datasets/SegmentationClass"
    classes = ["_background_","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    # 遍历待转换的 LabelMe JSON 标注。
    count = os.listdir("./datasets/before/")
    for i in range(0, len(count)):
        path = os.path.join("./datasets/before", count[i])

        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))

            # 优先读取 JSON 内嵌图像，否则从 imagePath 载入原图。
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)
            # 为当前标注中的类别分配连续索引。
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            # 按索引排序，确保局部类别编号连续。
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))

            # 将多边形标注栅格化为类别标签。
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            PIL.Image.fromarray(img).save(osp.join(jpgs_path, count[i].split(".")[0]+'.jpg'))

            # 将当前标注的局部索引转换为 classes 中的统一索引。
            new = np.zeros([np.shape(img)[0],np.shape(img)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                new = new + index_all*(np.array(lbl) == index_json)

            # 保存带调色板的 PNG；像素值仍为类别索引。
            utils.lblsave(osp.join(pngs_path, count[i].split(".")[0]+'.png'), new)
            print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')
