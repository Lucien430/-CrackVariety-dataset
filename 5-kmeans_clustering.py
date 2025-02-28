import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def process_images(input_folder, output_folder, n_clusters=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cluster_centers_list = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            filepath = os.path.join(input_folder, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # 将图像转换为一维数组
            pixels = image.reshape(-1, 1)

            # K-means聚类
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(pixels)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_

            # 将聚类结果转换为图像
            clustered_image = centroids[labels].reshape(image.shape).astype(np.uint8)

            # 保存聚类结果图像
            output_path = os.path.join(output_folder, f'{filename}_clustered.png')
            cv2.imwrite(output_path, clustered_image)

            # 绘制并保存聚类结果的直方图
            plt.figure()
            plt.hist(pixels[labels == 0], bins=256, range=(0, 256), color='gray', alpha=0.5,
                     label=f'Cluster 1 (center: {centroids[0][0]:.2f})')
            plt.hist(pixels[labels == 1], bins=256, range=(0, 256), color='blue', alpha=0.5,
                     label=f'Cluster 2 (center: {centroids[1][0]:.2f})')
            plt.xlabel('Gray Value')
            plt.ylabel('Frequency')
            plt.title(f'Cluster Histogram of {filename}')
            plt.legend()
            histogram_path = os.path.join(output_folder, f'{filename}_cluster_histogram.png')
            plt.savefig(histogram_path)
            plt.close()

            # 将聚类中心点添加到列表中
            cluster_centers_list.append({
                'filename': filename,
                'cluster_1': centroids[0][0],
                'cluster_2': centroids[1][0]
            })

    # 创建一个DataFrame来存储聚类中心点
    cluster_centers_df = pd.DataFrame(cluster_centers_list)

    # 计算聚类中心点的平均值
    cluster_centers_avg = cluster_centers_df[['cluster_1', 'cluster_2']].mean()

    # 保存聚类中心点和平均值到Excel文件
    excel_path = os.path.join(output_folder, 'cluster_centers.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        cluster_centers_df.to_excel(writer, index=False, sheet_name='Cluster Centers')
        cluster_centers_avg.to_frame(name='Average').transpose().to_excel(writer, index=False,
                                                                          sheet_name='Averages')

if __name__ == "__main__":
    input_folder = "D:/Desktop/yolo/yolov8/datasets/CrackVariety/3-gray_histogram"  # 输入图像文件夹路径
    output_folder = "D:/Desktop/yolo/yolov8/datasets/CrackVariety/4-kmeans_clustering"  # 输出文件夹路径
    process_images(input_folder, output_folder)