import os
import cv2
import numpy as np
from tqdm import tqdm

directory = 'data/batch1/split/rear/right'
files = os.listdir(directory)
print(len(files))

lower = np.array([0, 0, 0])
upper = np.array([33, 255, 255])

def show_image(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)

def get_mask(filepath, second_biggest=False):
    img = cv2.imread(filepath)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, lower, upper)

    # show_image('mask', mask)

    contours = cv2.findContours(mask, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours, key=cv2.contourArea, default=None)

    if second_biggest:
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[1]

    if cv2.contourArea(contour) < 30000 or contour is None:
        print("No leaf found!")

    cv2.drawContours(img, [contour], -1, (255, 255, 255), -1, cv2.LINE_AA)

    show_image('contours', img)

def main():
    for file in tqdm(files):
        get_mask(os.path.join(directory, file))

if __name__ == '__main__':
    main()

    FINAL_THRESHOLD = 33