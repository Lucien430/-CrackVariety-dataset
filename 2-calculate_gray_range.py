import os
import cv2
import numpy as np

def calculate_average_gray_range(input_folder):
    min_gray_values = []
    max_gray_values = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            filepath = os.path.join(input_folder, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # 计算每张图的灰度范围
            min_gray = np.min(image)
            max_gray = np.max(image)

            min_gray_values.append(min_gray)
            max_gray_values.append(max_gray)

            print(f'{filename}: Gray Range {min_gray}~{max_gray}')

    # 计算平均灰度范围
    average_min_gray = np.mean(min_gray_values)
    average_max_gray = np.mean(max_gray_values)

    print(f'Average Gray Range: {average_min_gray:.2f}~{average_max_gray:.2f}')

if __name__ == "__main__":
    input_folder = "C:/Users/Lucien/Desktop/yolo/yolov8/datasets/CFD/1-crop_images"  # 输入图像文件夹路径
    calculate_average_gray_range(input_folder)
