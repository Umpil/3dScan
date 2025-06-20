import os
import time
import cv2
import numpy as np
from graycode_work import get_codes, gray_to_decimal
import glob

image_folder = "SortedDataSet/Encoded"
lited_path = "SortedDataSet/Lited"
output_path = "SortedDataSet/Decoded"
gray_images = []
images = glob.glob(os.path.join(image_folder,'*.jpg'))
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_images.append(gray)

cache_h_codes = []
cache_v_codes = []
eps = 15
now_folder = f"SortedDataSet/Lited/WithoutOverflow{eps}"
if not os.path.exists(now_folder):
    os.mkdir(now_folder)
#h_codes, v_codes, h_images, v_images = get_codes(gray_images, ret_images=True, eps=eps)
h_codes, v_codes = get_codes(gray_images)
# for i, h_image in enumerate(h_images):
#     cv2.imwrite(now_folder + f"/h_{i}.jpg", h_image)

# for i, v_image in enumerate(v_images):
#     cv2.imwrite(now_folder + f"/v_{i}.jpg", v_image)

cache_h_codes.append(h_codes)
cache_v_codes.append(v_codes)
best_h_codes = np.max(cache_h_codes, axis=0)
best_v_codes = np.max(cache_v_codes, axis=0)

h_pixels = np.array([gray_to_decimal(best_h_codes[:, y, x])  for y in range(0, best_h_codes.shape[1]) for x in range(0, best_h_codes.shape[2])]).reshape((best_h_codes.shape[1], best_h_codes.shape[2]))
v_pixels = np.array([gray_to_decimal(np.flip(best_v_codes[:, y, x]))  for y in range(0, best_v_codes.shape[1])for x in range(0, best_v_codes.shape[2])] ).reshape((best_v_codes.shape[1], best_v_codes.shape[2]))

np.save(os.path.join(output_path, 'h_pixels.npy'), h_pixels)
np.save(os.path.join(output_path, 'v_pixels.npy'), v_pixels)
