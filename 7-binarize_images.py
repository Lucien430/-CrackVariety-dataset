import os
import cv2

def binarize_image(image, threshold=170):
    # 二值化处理
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            filepath = os.path.join(input_folder, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # 二值化处理
            binary_image = binarize_image(image)

            # 保存处理后的图像
            output_path = os.path.join(output_folder, f'{filename}')
            cv2.imwrite(output_path, binary_image)

if __name__ == "__main__":
    input_folder = "D:/Desktop/yolo/yolov8/datasets/CrackVariety/5-stretch_gray_values/val"  # 输入图像文件夹路径
    output_folder = "D:/Desktop/yolo/yolov8/datasets/CrackVariety/6-binarize_images-170/val"  # 输出文件夹路径
    process_images(input_folder, output_folder)
