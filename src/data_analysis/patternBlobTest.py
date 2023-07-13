import os
import cv2
import numpy as np
from tqdm import tqdm

TEST_IMAGE = 'data\\split\\arm\\front\\right\\CAM046068_d.jpg'

lower = np.array([0, 0, 0])
upper = np.array([25, 255, 255])

def show_image(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    
def find_blobs(img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, lower, upper)

    contours = cv2.findContours(mask, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours, key=cv2.contourArea, default=None)

    if contour is None or cv2.contourArea(contour) < 50000:
        # print("No leaf found!")
        return []

    mask = np.zeros_like(mask, dtype=np.uint8)

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    result = cv2.bitwise_and(img, img, mask=mask)

    img_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, lower, upper)

    contours = cv2.findContours(mask, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)[0]

    result = cv2.bitwise_and(result, result, mask=mask)

    x, y, w, h = cv2.boundingRect(contour)

    detector = cv2.SimpleBlobDetector_create()

    keypoints = detector.detect(result)

    show_image(
        "Keypoints",
        cv2.drawKeypoints(result, keypoints, np.array([]), (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

    for keypoint in keypoints:
        keypoint.pt = (keypoint.pt[0] - x, keypoint.pt[1] - y)
        keypoint.pt = (keypoint.pt[0] / w, keypoint.pt[1] / h)

    return keypoints

minBlobs = 100000
minFile = ""

cwd = os.getcwd()

folderArray = [(i, j, k) for i in ["arm", "zea"] for j in ["front", "rear"]
               for k in ["left", "right"]]

print(folderArray)

for folder in tqdm(folderArray):

    folder = os.path.join(cwd, "data", "split", folder[0], folder[1],
                          folder[2])

    files = list(filter(lambda x: x.endswith('.jpg'), os.listdir(folder)))

    for file in tqdm(files, leave=False):
        fileKeypoints = find_blobs(cv2.imread(os.path.join(folder, file)))

        if len(fileKeypoints) < minBlobs:
            minBlobs = len(fileKeypoints)
            minFile = os.path.join(folder, file)

print(minBlobs)
print(minFile)

# find_blobs(TEST_IMAGE)