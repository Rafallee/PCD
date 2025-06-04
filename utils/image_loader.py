import cv2
import os

def load_images_from_folder(folder):
    images = []
    labels = []
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        for filename in os.listdir(class_path):
            path = os.path.join(class_path, filename)
            images.append(path)
            labels.append(class_name)
    return images, labels