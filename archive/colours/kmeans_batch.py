import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

cwd = os.getcwd()
folder = os.path.join(cwd, 'data', "unlabelled_batch1", "split", "rear", "right")

# files = list(filter(lambda x: x.endswith('.jpg'), os.listdir(folder)))
files = os.listdir(folder)

lower = np.array([0, 0, 0])
upper = np.array([25, 255, 255])

def show_image(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)

with open('kmeans_results.csv', 'w', newline='', encoding='UTF8') as csvfile:
    writer = csv.writer(csvfile)

    columns = ["filename"] + \
        [f"colour_{i}_{j}" for i in range(4) for j in ["r", "g", "b"]] + \
        [f"percentage_{i}" for i in range(4)]

    writer.writerow(columns)

    for file in tqdm(files):
        filepath = os.path.join(folder, file)
        img = cv2.imread(filepath)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(img_hsv, lower, upper)

        contours = cv2.findContours(mask, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        contour = max(contours, key=cv2.contourArea, default=None)

        if contour is None or cv2.contourArea(contour) < 50000:
            # print("No leaf found!")
            writer.writerow([file])
            continue

        mask = np.zeros_like(mask, dtype=np.uint8)

        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

        result = cv2.bitwise_and(img, img, mask=mask)

        img_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(img_hsv, lower, upper)

        contours = cv2.findContours(mask, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]

        result = cv2.bitwise_and(result, result, mask=mask)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result = result.reshape((result.shape[0] * result.shape[1], 3))
        result = result[result.sum(axis=1) > 0]

        kmeans = KMeans(n_clusters=4, random_state=0,
                        n_init='auto').fit(result)

        colours = kmeans.cluster_centers_

        percentages = np.bincount(kmeans.labels_) / kmeans.labels_.shape[0]

        row = [file] + list(colours.flatten()) + list(percentages.flatten())

        writer.writerow(row)