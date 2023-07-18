import os
from enum import Enum
import cv2
import numpy as np
import feret

class Batch(Enum):
    ONE = 1846.0531 / 40
    TWO = 1406.0228 / 40

TEST_IMAGE = 'data/sample_test_images/test2.jpg'
DEFAULT_BATCH = Batch.ONE
SHOW_IMAGES = False

lower = np.array([0, 0, 0])
upper = np.array([33, 255, 255])

def show_image(name, image):
    if SHOW_IMAGES:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, image)
        cv2.waitKey(0)

def underlined(text):
    return "\033[4m" + text + "\033[0m"

def bold(text):
    return "\033[1m" + text + "\033[0m"

def find_contour(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, lower, upper)

    contours = cv2.findContours(mask, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours, key=cv2.contourArea, default=None)

    return mask, contour

def analyse_as_string(image, BATCH, print_results=True):
    area, major, minor, circ, maxFeret, minFeret, ar = analyse(
        image, BATCH, print_results)
    
    area = str(area or "")
    major = str(major or "")
    minor = str(minor or "")
    circ = str(circ or "")
    maxFeret = str(maxFeret or "")
    minFeret = str(minFeret or "")
    ar = str(ar or "")

    return area, major, minor, circ, maxFeret, minFeret, ar

def analyse(image, BATCH, print_results=True):

    img = cv2.imread(image)

    mask, contour = find_contour(img)

    if contour is None:
        return None, None, None, None, None, None, None

    # make binary image from contour
    binary = np.zeros_like(mask)
    cv2.drawContours(binary, [contour], -1, (255, 255, 255), -1, cv2.LINE_AA)

    # show_image('contours', img)

    ellipse = cv2.fitEllipse(contour)

    cv2.ellipse(img, ellipse, (255, 255, 255), 2, cv2.LINE_AA)

    # show_image('ellipse', img)

    major = max(ellipse[1][0], ellipse[1][1]) / BATCH.value
    minor = min(ellipse[1][0], ellipse[1][1]) / BATCH.value

    pixelArea = cv2.contourArea(contour)
    area = pixelArea / (BATCH.value**2)

    circ = 4 * np.pi * pixelArea / (cv2.arcLength(contour, True)**2)

    ar = major / minor

    maxFeret = feret.max(binary) / BATCH.value
    minFeret = feret.min(binary) / BATCH.value

    if print_results:
        print(bold(underlined(os.path.basename(image))))
        print("Area:", area)
        print("Major axis:", major)
        print("Minor axis:", minor)
        print("Circularity:", circ)
        print("Max Feret:", maxFeret)
        print("Min Feret:", minFeret)
        print("Aspect ratio:", ar)
        print()

    # cv2.destroyAllWindows()

    return area, major, minor, circ, maxFeret, minFeret, ar

if __name__ == '__main__':
    # analyse(TEST_IMAGE)
    analyse(r'data\batch1\split\front\left\CAM046001_d.jpg', BATCH=Batch.ONE)
    analyse(r'data\batch1\split\front\left\CAM046004_d.jpg', BATCH=Batch.ONE)
