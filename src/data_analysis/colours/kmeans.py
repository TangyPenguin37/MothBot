import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

TEST_IMAGE = 'data/batch1/split/rear/left/CAM046086_v.jpg'

lower = np.array([0, 0, 0])
upper = np.array([37, 255, 255])

def show_image(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    
def get_kmeans(filepath, n_clusters=4, second_biggest=False):
    img = cv2.imread(filepath)

    show_image('img', img)

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
        return None, None

    mask = np.zeros_like(mask, dtype=np.uint8)

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    result = cv2.bitwise_and(img, img, mask=mask)

    show_image('result', result)

    img_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, lower, upper)

    contours = cv2.findContours(mask, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)[0]

    result = cv2.bitwise_and(result, result, mask=mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = result.reshape((result.shape[0] * result.shape[1], 3))
    result = result[result.sum(axis=1) > 0]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0,
                    n_init='auto').fit(result)

    colours = kmeans.cluster_centers_

    percentages = np.bincount(kmeans.labels_) / kmeans.labels_.shape[0]

    print(percentages)
    print(colours)

    return colours, percentages

def test():
    test_colours, test_percentages = get_kmeans(TEST_IMAGE)

    if test_colours is None or test_percentages is None:
        return

    cumulative = np.cumsum(test_percentages)
    cumulative = np.insert(cumulative, 0, 0)

    rectangles = np.zeros((100, 400, 3), dtype=np.uint8)

    for i, colour in enumerate(test_colours):
        cv2.rectangle(
            rectangles,
            (int(cumulative[i] * 400), 0, int(cumulative[i + 1] * 400), 100),
            (int(colour[2]), int(colour[1]), int(colour[0])), -1)

    # show_image('rectangles', rectangles)

def main():
    cwd = os.getcwd()
    folder = os.path.join(cwd, "data", "batch2", "split", "rear", "right")

    # files = list(filter(lambda x: x.endswith('.jpg'), os.listdir(folder)))
    files = os.listdir(folder)

    with open('kmeans_results.csv', 'w', newline='',
              encoding='UTF8') as csvfile:
        writer = csv.writer(csvfile)

        columns = ["filename"] + \
            [f"colour_{i}_{j}" for i in range(4) for j in ["r", "g", "b"]] + \
            [f"percentage_{i}" for i in range(4)]

        writer.writerow(columns)

        for file in tqdm(files):
            filepath = os.path.join(folder, file)

            colours, percentages = get_kmeans(filepath)

            if colours is None or percentages is None:
                writer.writerow([file])
                continue

            row = [file] + list(colours.flatten()) + list(
                percentages.flatten())

            writer.writerow(row)

if __name__ == "__main__":
    main()