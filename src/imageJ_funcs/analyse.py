from enum import Enum
import cv2
import numpy as np

class Batch(Enum):
    ONE = 1846.0531 / 40
    TWO = 1406.0228 / 40

TEST_IMAGE = 'data/test_images/test2.jpg'
BATCH = Batch.ONE

print(BATCH.value)

lower = np.array([0, 0, 0])
upper = np.array([33, 255, 255])

def show_image(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)

def analyse(image):
    img = cv2.imread(image)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, lower, upper)

    contours = cv2.findContours(mask, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours, key=cv2.contourArea, default=None)

    if cv2.contourArea(contour) < 20000 or contour is None:
        print("No leaf found!")

    # cv2.drawContours(img, [contour], -1, (255, 255, 255), 2, cv2.LINE_AA)

    # show_image('contours', img)

    ellipse = cv2.fitEllipse(contour)

    cv2.ellipse(img, ellipse, (255, 255, 255), 2, cv2.LINE_AA)  # type: ignore

    show_image('ellipse', img)

    major = max(ellipse[1][0], ellipse[1][1]) / BATCH.value
    minor = min(ellipse[1][0], ellipse[1][1]) / BATCH.value

    area = cv2.contourArea(contour) / (BATCH.value**2)

    print("Major axis: ", major)
    print("Minor axis: ", minor)
    print("Area: ", area)

    # cv2.destroyAllWindows()

    return major, minor, area

analyse(TEST_IMAGE)