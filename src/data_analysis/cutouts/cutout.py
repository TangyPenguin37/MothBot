### Crops images to the bounding box of the wing and removes the background ###

import os
import cv2
import numpy as np
from tqdm import tqdm

imagesDir = r'data\split_data\train\images\all\split_filtered'
cutoutDir = r'data\split_data\train\images\all\cutout'

fileWalk = os.walk(imagesDir)
allFiles = []

batch1_rear_right = [
    "CAM046070_d", "CAM046114_d", "CAM046114_v", "CAM046126_d", "CAM046126_v",
    "CAM046142_d", "CAM046227_d", "CAM046227_v", "CAM046258_v", "CAM046259_d",
    "CAM046259_v", "CAM046263_d", "CAM046263_v", "CAM046264_d", "CAM046264_v"
]

batch2_rear_left = [
    "CAM046331_d", "CAM046331_v", "CAM046939_d", "CAM046939_v", "CAM047018_d",
    "CAM047018_v", "CAM047227_d", "CAM047399_d", "CAM047401_d", "CAM047401_v",
    "CAM047606_d", "CAM047606_v", "CAM047607_d", "CAM047607_v", "CAM047610_d",
    "CAM047610_v,", "CAM047612_d", "CAM047612_v", "CAM047614_d", "CAM047614_v",
    "CAM047615_d", "CAM047615_v", "CAM047620_d", "CAM047620_v"
]

def show_image(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)

for root, dirs, files in fileWalk:
    for file in files:
        filepath = os.path.join(root, file)
        filepath = os.path.join(*filepath.split(os.sep)[-4:])
        allFiles.append(filepath)

for file in tqdm(allFiles):
    filepath = os.path.join(imagesDir, file)
    img = cv2.imread(filepath)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([33, 255, 255]))

    contours = cv2.findContours(mask, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours, key=cv2.contourArea, default=None)

    bounding_box = cv2.boundingRect(contour)

    mask = np.zeros_like(mask, dtype=np.uint8)

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    result = cv2.bitwise_and(img, img, mask=mask)

    x, y, w, h = bounding_box
    result = result[y:y + h, x:x + w]

    # show_image('result', result)

    fileFolder = os.path.dirname(os.path.join(cutoutDir, file))

    if not os.path.exists(fileFolder):
        os.makedirs(fileFolder)

    if not cv2.imwrite(os.path.join(cutoutDir, file), result):
        raise RuntimeError(f"Could not write image {file} to file!")