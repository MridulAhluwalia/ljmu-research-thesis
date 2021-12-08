import os

import PIL
from PIL import Image

import cv2
import numpy as np

ANALYSIS_PATH = (
    "/Users/mac/Projects/learning/LJMU/ljmu-research-thesis/analysis"
)
model = "PGGAN_C"
num = 14
size = 128
IMG_PATH = f"{model}/{num}"
IMG_LIST = [f"inception_image_size_{size}.png", f"ssid_image_size_{size}.png"]
list_im = [os.path.join(ANALYSIS_PATH, IMG_PATH, img) for img in IMG_LIST]
print(list_im)

imgs = [PIL.Image.open(i) for i in list_im]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

# save that beautiful picture
imgs_comb = PIL.Image.fromarray(imgs_comb)
imgs_comb.save(os.path.join(ANALYSIS_PATH, IMG_PATH, f"eval_{size}.png"))
