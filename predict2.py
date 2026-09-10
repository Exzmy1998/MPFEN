"""MPFEN 批量推理入口：支持二值掩码输出及视频、FPS、ONNX 模式。

单图和目录模式均以 200 × 200 像素分块推理，再拼接二值裂缝掩码。
"""

import time
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from mpfen import MPFEN

if __name__ == "__main__":
    # 创建推理器；类别颜色可在 MPFEN.__init__ 中配置。
    mpfen = MPFEN(model_path="logs/best_epoch_weights.pth")

    # 模式：predict、video、fps、dir_predict、export_onnx；图像模式保存二值掩码。
    mode = "dir_predict"

    # 图像模式：控制类别像素计数及输出名称。
    count = False
    name_classes = ["background", "crack"]

    # 视频模式：路径为 0 时使用摄像头；保存路径为空时不写入视频。
    video_path = 0
    video_save_path = ""
    video_fps = 25.0

    # FPS 模式：重复预测同一图像，统计平均推理耗时。
    test_interval = 100
    fps_image_path = "img/street.jpg"

    # 目录模式：输入图像目录与二值掩码目录。
    dir_origin_path = "img/matched_jpgs/"
    dir_save_path = "img/matched_pngs/"

    # ONNX 导出：是否简化计算图，以及模型保存路径。
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":
        # 使用纯分割图，以类别颜色提取二值裂缝掩码。
        original_mix_type = mpfen.mix_type
        mpfen.mix_type = 1

        # 分块尺寸为 200 × 200；边缘块保留实际尺寸。
        BLOCK_SIZE = 200
        # 裂缝颜色须与 MPFEN.colors 中索引 1 的颜色一致。
        crack_color = (128, 0, 0)

        while True:
            img_path = input('Input image filename:')
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print('Open Error! Try again! Error:', e)
                continue

            w, h = image.size
            output_img = Image.new("L", (w, h), 0)

            # 逐块预测并拼接，裂缝像素输出为 255。
            for top in range(0, h, BLOCK_SIZE):
                for left in range(0, w, BLOCK_SIZE):
                    box = (left, top, min(left + BLOCK_SIZE, w), min(top + BLOCK_SIZE, h))
                    patch = image.crop(box)

                    pred_patch = mpfen.detect_image(patch, count=count, name_classes=name_classes)
                    if not isinstance(pred_patch, Image.Image):
                        pred_patch = Image.fromarray(pred_patch)

                    pred_np = np.array(pred_patch)
                    # 将裂缝类别颜色映射为白色，其余像素映射为黑色。
                    binary_mask = np.all(pred_np == crack_color, axis=-1).astype(np.uint8) * 255
                    binary_patch = Image.fromarray(binary_mask, mode='L')

                    output_img.paste(binary_patch, box)

            output_img.show()

            # 二值掩码以 PNG 格式保存到 img/predict。
            img_filename = os.path.basename(img_path)
            base, ext = os.path.splitext(img_filename)
            save_filename = base + ".png"
            save_path = os.path.join("img/predict", save_filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            output_img.save(save_path)
            print('Save to', save_path)

        # 恢复进入该模式前的显示配置。
        mpfen.mix_type = original_mix_type

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while True:
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            # OpenCV 帧转换为 RGB 图像后送入推理器。
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(mpfen.detect_image(frame))
            # 转回 BGR，以便 OpenCV 显示及写入视频。
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = mpfen.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        # 使用纯分割图，以类别颜色提取二值裂缝掩码。
        original_mix_type = mpfen.mix_type
        mpfen.mix_type = 1

        # 分块尺寸为 200 × 200；边缘块保留实际尺寸。
        BLOCK_SIZE = 200
        # 裂缝颜色须与 MPFEN.colors 中索引 1 的颜色一致。
        crack_color = (128, 0, 0)

        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        img_names = os.listdir(dir_origin_path)
        print(f"Start processing images from '{dir_origin_path}' to '{dir_save_path}'...")

        # 遍历输入目录中的图像文件。
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)

                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f'Open {img_name} Error! Error:', e)
                    continue

                w, h = image.size
                # 创建与原图等大的灰度画布，背景像素为 0。
                output_img = Image.new("L", (w, h), 0)

                # 逐块预测并拼接，裂缝像素输出为 255。
                for top in range(0, h, BLOCK_SIZE):
                    for left in range(0, w, BLOCK_SIZE):
                        box = (left, top, min(left + BLOCK_SIZE, w), min(top + BLOCK_SIZE, h))
                        patch = image.crop(box)

                        pred_patch = mpfen.detect_image(patch, count=count, name_classes=name_classes)
                        if not isinstance(pred_patch, Image.Image):
                            pred_patch = Image.fromarray(pred_patch)

                        pred_np = np.array(pred_patch)
                        # 将裂缝类别颜色映射为白色，其余像素映射为黑色。
                        binary_mask = np.all(pred_np == crack_color, axis=-1).astype(np.uint8) * 255
                        binary_patch = Image.fromarray(binary_mask, mode='L')

                        output_img.paste(binary_patch, box)

                # 统一保存为 PNG，避免有损压缩改变二值掩码。
                base, ext = os.path.splitext(img_name)
                save_filename = base + ".png"
                output_img.save(os.path.join(dir_save_path, save_filename))

        # 恢复进入该模式前的显示配置。
        mpfen.mix_type = original_mix_type
        print("All images processed successfully!")

    elif mode == "export_onnx":
        mpfen.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
