import cv2
import pyefd
import numpy as np
from sklearn.cluster import KMeans

lower = np.array([0, 0, 0])
upper = np.array([33, 255, 255])

def get_kmeans(filepath, n_clusters=2):

    img = cv2.imread(filepath)

    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower, upper)

    contour = max(cv2.findContours(mask, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)[0],
                  key=cv2.contourArea,
                  default=None)

    if cv2.contourArea(contour) < 20000 or contour is None:
        print("No wing found!")
        return None, None, None

    mask = np.zeros_like(mask, dtype=np.uint8)

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    result = cv2.bitwise_and(img, img, mask=mask)
    result = cv2.bitwise_and(result,
                             result,
                             mask=cv2.inRange(
                                 cv2.cvtColor(result, cv2.COLOR_BGR2HSV),
                                 lower, upper))
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = result.reshape((result.shape[0] * result.shape[1], 3))
    result = result[result.sum(axis=1) > 0]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0,
                    n_init='auto').fit(result)

    colours = kmeans.cluster_centers_

    percentages = np.bincount(kmeans.labels_) / kmeans.labels_.shape[0]

    return colours, percentages

def get_efd(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2HSV)

    contour = max(cv2.findContours(cv2.inRange(img, lower, upper),
                                   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0],
                  key=cv2.contourArea,
                  default=None)

    if cv2.contourArea(contour) < 20000 or contour is None:
        print("No wing found!")
        return None, None

    coeffs = pyefd.elliptic_fourier_descriptors(
        np.squeeze(contour), order=7,
        normalize=True).flatten()[3:]  # type: ignore

    return coeffs