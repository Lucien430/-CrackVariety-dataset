import os
import cv2

def crop_images(input_folder, output_folder, crop_size=(128, 99)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            filepath = os.path.join(input_folder, filename)
            image = cv2.imread(filepath)

            img_height, img_width = image.shape[:2]
            crop_height, crop_width = crop_size

            for i in range(0, img_height, crop_height):
                for j in range(0, img_width, crop_width):
                    crop_img = image[i:i+crop_height, j:j+crop_width]
                    crop_filename = f"{filename.split('.')[0]}_{i}_{j}.png"
                    crop_filepath = os.path.join(output_folder, crop_filename)
                    cv2.imwrite(crop_filepath, crop_img)

if __name__ == "__main__":
    input_folder = "D:/Desktop/yolo/yolov8/datasets/CrackVariety/images"  # 输入图像文件夹路径
    output_folder = "D:/Desktop/yolo/yolov8/datasets/CrackVariety/1-crop_images"  # 输出文件夹路径
    crop_images(input_folder, output_folder)
