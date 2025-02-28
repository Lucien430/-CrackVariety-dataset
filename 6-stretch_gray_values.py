import os
import cv2
import numpy as np

def stretch_gray_values(image, lower_bound=6.74, upper_bound=254.07):
    # 将图像的灰度值从0~255线性拉伸到lower_bound~upper_bound
    stretched_image = np.interp(image, (image.min(), image.max()), (lower_bound, upper_bound))
    return stretched_image.astype(np.uint8)

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            filepath = os.path.join(input_folder, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # 拉伸灰度值
            stretched_image = stretch_gray_values(image)

            # 保存处理后的图像
            output_path = os.path.join(output_folder, f'{filename}')
            cv2.imwrite(output_path, stretched_image)

if __name__ == "__main__":
    input_folder = "D:/Desktop/yolo/yolov8/datasets/CrackVariety/images"  # 输入图像文件夹路径
    output_folder = "D:/Desktop/yolo/yolov8/datasets/CrackVariety/5-stretch_gray_values/train"  # 输出文件夹路径
    process_images(input_folder, output_folder)
