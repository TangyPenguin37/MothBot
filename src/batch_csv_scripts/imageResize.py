import os
import cv2
import numpy as np
from tqdm import tqdm

width = 1024
height = 1024

cutoutDir = r'data\split_data\train\images\all\cutout'
newCutoutDir = r'data\split_data\train\images\all\cutout_resized'

def show_image(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)

if not os.path.exists(newCutoutDir):
    os.mkdir(newCutoutDir)

for filepath in tqdm([
        os.path.join(root, file) for root, _, files in os.walk(cutoutDir)
        for file in files
]):
    img = cv2.imread(filepath)
    # pad image to square of 1024x1024
    paddingTop = (height - img.shape[0]) // 2
    paddingBottom = height - img.shape[0] - paddingTop
    paddingLeft = (width - img.shape[1]) // 2
    paddingRight = width - img.shape[1] - paddingLeft

    img = cv2.copyMakeBorder(img, paddingTop, paddingBottom, paddingLeft,
                             paddingRight, cv2.BORDER_CONSTANT)

    assert img.shape == (height, width, 3)

    fileFolder = os.path.join(*filepath.split(os.sep)[-4:])

    if not os.path.exists(
            os.path.join(newCutoutDir, os.path.dirname(fileFolder))):
        os.makedirs(os.path.join(newCutoutDir, os.path.dirname(fileFolder)))

    assert cv2.imwrite(os.path.join(newCutoutDir, fileFolder), img) is True
