import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def process_images(input_folder, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 创建一个DataFrame来存储灰度值分布
    all_histograms = pd.DataFrame()
    total_hist = np.zeros(256, dtype=int)  # 初始化总的灰度直方图

    # 遍历输入文件夹中的所有图像文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            filepath = os.path.join(input_folder, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # 计算灰度直方图
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = hist.flatten().astype(int)  # 将直方图转换为int类型

            # 获取图像的实际灰度值范围，且该范围内的值出现次数大于等于
            min_gray = np.min(np.where(hist >= 1)[0])
            max_gray = np.max(np.where(hist >= 1)[0])

            # 累加到总的灰度直方图
            total_hist += hist

            # 将直方图数据添加到DataFrame中
            all_histograms[filename] = hist

            # 绘制灰度直方图
            plt.figure()
            plt.bar(range(min_gray, max_gray + 1), hist[min_gray:max_gray + 1], width=1, color='gray')
            plt.xlabel('Gray')
            plt.ylabel('Frequency')
            plt.title(f'Gray Histogram of {filename}')
            plt.xlim([min_gray, max_gray])

            # 保存灰度直方图
            histogram_path = os.path.join(output_folder, f'{filename}_histogram.png')
            plt.savefig(histogram_path)
            plt.close()

    # 绘制和保存总的灰度直方图
    total_min_gray = np.min(np.where(total_hist >= 1)[0])
    total_max_gray = np.max(np.where(total_hist >= 1)[0])
    plt.figure()
    plt.bar(range(total_min_gray, total_max_gray + 1), total_hist[total_min_gray:total_max_gray + 1],
            width=1, color='gray')
    plt.xlabel('Gray')
    plt.ylabel('Frequency')
    plt.title('Total Gray Histogram')
    plt.xlim([total_min_gray, total_max_gray])

    total_histogram_path = os.path.join(output_folder, 'total_histogram.png')
    plt.savefig(total_histogram_path)
    plt.close()

    # 保存所有图像的灰度值分布到Excel文件
    excel_path = os.path.join(output_folder, 'gray_histograms.xlsx')
    all_histograms.to_excel(excel_path, index=False)

if __name__ == "__main__":
    input_folder = "D:/Desktop/yolo/yolov8/datasets/CrackVariety/2-stretch_gray_values"  # 输入图像文件夹路径
    output_folder = "D:/Desktop/yolo/yolov8/datasets/CrackVariety/3-gray_histogram"  # 输出文件夹路径
    process_images(input_folder, output_folder)
