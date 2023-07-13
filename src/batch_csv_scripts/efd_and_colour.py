import os
import csv
import cv2
import numpy as np
import pyefd
from tqdm import tqdm
from sklearn.cluster import KMeans

def show_image(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)

cwd = os.getcwd()
folder = os.path.join(cwd, 'data', "batch2", "split", "tofix", "rearright", "set2")

files = list(filter(lambda x: x.endswith('.jpg'), os.listdir(folder)))

with open('results.csv', 'w', newline='', encoding='UTF8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        ["filename"] + ["efd_" + str(i) for i in range(25)] +
        [f"colour_{i}_{j}" for i in range(4)
         for j in ["r", "g", "b"]] + [f"percentage_{i}" for i in range(4)])

    for file in tqdm(files):
        filepath = os.path.join(folder, file)
        img = cv2.imread(filepath)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 0, 0])
        upper = np.array([35, 255, 255])

        mask = cv2.inRange(img_hsv, lower, upper)

        contours = cv2.findContours(mask, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        contour = max(contours, key=cv2.contourArea, default=None)

        # secondbiggest
        # contour = sorted(contours, key=cv2.contourArea, reverse=True)[1]

        if cv2.contourArea(contour) < 20000 or contour is None:
            print("No leaf found!")
            coeffs = [""] * 25
        else:
            cv2.drawContours(img, [contour], -1, (255, 255, 255), 2,
                             cv2.LINE_AA)

            # show_image("contour", img)

            efd_coeff = pyefd.elliptic_fourier_descriptors(np.squeeze(contour),
                                                           order=7,
                                                           normalize=True)

            coeffs = efd_coeff.flatten()[3:].tolist() # type: ignore

        # COLOUR

        img = cv2.imread(filepath)

        # show_image("img", img)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 0, 0])
        upper = np.array([35, 255, 255])

        mask = cv2.inRange(img_hsv, lower, upper)

        contours = cv2.findContours(mask, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        contour = max(contours, key=cv2.contourArea, default=None)

        # secondbiggest
        # contour = sorted(contours, key=cv2.contourArea, reverse=True)[1]

        if contour is None or cv2.contourArea(contour) < 20000:
            colours = [""] * 12 + [""] * 4

        else:
            mask = np.zeros_like(mask, dtype=np.uint8)

            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

            result = cv2.bitwise_and(img, img, mask=mask)

            img_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(img_hsv, lower, upper)

            contours = cv2.findContours(mask, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)[0]

            result = cv2.bitwise_and(result, result, mask=mask)

            # show_image("result", result)

            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result = result.reshape((result.shape[0] * result.shape[1], 3))
            result = result[result.sum(axis=1) > 0]

            kmeans = KMeans(n_clusters=4, random_state=0,
                            n_init='auto').fit(result)

            colours = kmeans.cluster_centers_

            percentages = np.bincount(kmeans.labels_) / kmeans.labels_.shape[0]

            # row = [file] + list(colours.flatten()) + list(percentages.flatten())

            # writer.writerow(row)

            colours = colours.flatten().tolist() + percentages.tolist()

        writer.writerow([file] + coeffs + colours)