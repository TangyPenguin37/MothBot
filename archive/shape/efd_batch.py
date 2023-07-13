import os
import csv
import cv2
import numpy as np
import pyefd
from tqdm import tqdm

cwd = os.getcwd()
folder = os.path.join(cwd, 'data', "labelled", "split_corrected",
                      "zea", "rear", "right")

files = list(filter(lambda x: x.endswith('.jpg'), os.listdir(folder)))

with open('py_results.csv', 'w', newline='', encoding='UTF8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename"] + ["efd_" + str(i) for i in range(25)])

    for file in tqdm(files):
        filepath = os.path.join(folder, file)
        img = cv2.imread(filepath)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 0, 0])
        upper = np.array([33, 255, 255])

        mask = cv2.inRange(img_hsv, lower, upper)

        contours = cv2.findContours(mask, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        contour = max(contours, key=cv2.contourArea, default=None)

        if cv2.contourArea(contour) < 50000 or contour is None:
            # print("No leaf found!")
            writer.writerow([file])
            continue

        # print(cv2.contourArea(contour))

        cv2.drawContours(img, [contour], -1, (255, 255, 255), 2, cv2.LINE_AA)

        efd_coeff = pyefd.elliptic_fourier_descriptors(np.squeeze(contour),
                                                       order=7,
                                                       normalize=True)

        coeffs = efd_coeff.flatten()[3:].tolist()  # type: ignore

        writer.writerow([file] + coeffs)