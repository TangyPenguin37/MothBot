import os
import sys
import csv
import cv2
import numpy as np
import pyefd
from tqdm import tqdm

TEST_IMAGE = 'data/batch1/split/rear/left/CAM046086_v.jpg'

lower = np.array([0, 0, 0])
upper = np.array([33, 255, 255])

def show_image(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    
def efd(filepath, second_biggest=False):
    img = cv2.imread(filepath)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, lower, upper)

    show_image('mask', mask)

    contours = cv2.findContours(mask, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours, key=cv2.contourArea, default=None)

    if second_biggest:
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[1]

    if cv2.contourArea(contour) < 20000 or contour is None:
        print("No leaf found!")
        return None

    cv2.drawContours(img, [contour], -1, (255, 255, 255), -1, cv2.LINE_AA)

    show_image('contours', img)

    efd_coeff = pyefd.elliptic_fourier_descriptors(np.squeeze(contour),
                                                   order=7,
                                                   normalize=True)

    coeffs = efd_coeff.flatten()[3:]  # type: ignore

    return coeffs

def test():

    print(efd(TEST_IMAGE))

def main():

    cwd = os.getcwd()
    folder = os.path.join(cwd, "data", "batch2", "split", "rear", "right")

    files = list(filter(lambda x: x.endswith('.jpg'), os.listdir(folder)))

    with open('efd_results.csv', 'w', newline='', encoding='UTF8') as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(["filename"] + ["efd_" + str(i) for i in range(25)])

        for file in tqdm(files):

            filepath = os.path.join(folder, file)

            coeffs = efd(filepath)

            if coeffs is None:
                writer.writerow([file])
                continue

            writer.writerow([file] + coeffs)

if __name__ == '__main__':
    main()