# === ekstraksi/warna.py ===
import cv2
import numpy as np
import os

def ekstrak_warna(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (200, 200))
    chans = cv2.split(image)
    features = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [32], [0, 256])
        features.extend(hist.flatten())
    return np.array(features)
