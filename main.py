# === main.py ===
import os
import csv
from ekstraksi import warna, bentuk, tekstur
from utils.image_loader import load_images_from_folder

output_folder = "fitur_dataset"
os.makedirs(output_folder, exist_ok=True)

def simpan_fitur(fitur_fn, fitur_list, labels):
    with open(fitur_fn, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f'f{i}' for i in range(len(fitur_list[0]))] + ['label']
        writer.writerow(header)
        for feat, lbl in zip(fitur_list, labels):
            writer.writerow(list(feat) + [lbl])

folder_dataset = 'data/original'
images, labels = load_images_from_folder(folder_dataset)

# Warna
fitur_warna = [warna.ekstrak_warna(img) for img in images]
simpan_fitur(os.path.join(output_folder, 'fitur_warna.csv'), fitur_warna, labels)

# Bentuk
fitur_bentuk = [bentuk.ekstrak_bentuk(img) for img in images]
simpan_fitur(os.path.join(output_folder, 'fitur_bentuk.csv'), fitur_bentuk, labels)

# Tekstur
fitur_tekstur = [tekstur.ekstrak_tekstur(img) for img in images]
simpan_fitur(os.path.join(output_folder, 'fitur_tekstur.csv'), fitur_tekstur, labels)

print("Ekstraksi fitur selesai.")
