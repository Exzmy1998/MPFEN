"""生成 VOC 数据集划分文件，并检查标签格式与类别像素数量。

默认不保留独立测试集，训练集和验证集按 9:1 划分。
"""

import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

# trainval_percent 控制训练与验证样本占比，剩余样本划入测试集。
trainval_percent = 1
# train_percent 控制训练集在训练与验证样本中的占比。
train_percent = 0.9

# VOC 数据集根目录。
VOCdevkit_path = 'VOCdevkit'

if __name__ == "__main__":
    # 固定随机种子，使同一文件列表的划分可复现。
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath = os.path.join(VOCdevkit_path, 'Test0717/SegmentationClass')
    saveBasePath = os.path.join(VOCdevkit_path, 'Test0717/ImageSets/Segmentation')

    # 以 PNG 标签文件为依据收集样本。
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    # 按配置比例采样训练、验证和测试集索引。
    num = len(total_seg)
    list = range(num)
    tv = int(num*trainval_percent)
    tr = int(tv*train_percent)
    trainval = random.sample(list,tv)
    train = random.sample(trainval,tr)

    print("train and val size",tv)
    print("traub suze",tr)
    # 写入不带扩展名的样本 ID，每行一个。
    ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath,'val.txt'), 'w')

    for i in list:
        name = total_seg[i][:-4]+'\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")
    # 统计 0～255 标签值的像素数量，并检查标签维度。
    classes_nums = np.zeros([256], np.int)
    for i in tqdm(list):
        name = total_seg[i]
        png_file_name = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。"%(png_file_name))

        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。"%(name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。"%(name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)

    # 二分类标签应使用 0（背景）和 1（目标），而非显示用的 0/255。
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print("JPEGImages中的图片应当为.jpg文件、SegmentationClass中的图片应当为.png文件。")
    print("如果格式有误，参考:")
    print("https://github.com/bubbliiiing/segmentation-format-fix")
