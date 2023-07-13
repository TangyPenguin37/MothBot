import sys
import cv2
import numpy as np
import pyefd

TEST_IMAGE = 'data/batch1/split/rear/left/CAM046086_v.jpg'

def show_image(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    
img = cv2.imread(TEST_IMAGE)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([0, 0, 0])
upper = np.array([37, 255, 255])

mask = cv2.inRange(img_hsv, lower, upper)

show_image('mask', mask)

contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
contour = max(contours, key=cv2.contourArea, default=None)

# second biggest
contour = sorted(contours, key=cv2.contourArea, reverse=True)[1]

if cv2.contourArea(contour) < 20000 or contour is None:
    print("No leaf found!")
    sys.exit()

cv2.drawContours(img, [contour], -1, (255, 255, 255), 2, cv2.LINE_AA)

show_image('contours', img)

efd_coeff = pyefd.elliptic_fourier_descriptors(np.squeeze(contour),
                                               order=7,
                                               normalize=True)

# dc_coeffs = (pyefd.calculate_dc_coefficients(np.squeeze(contour))[1],
#              pyefd.calculate_dc_coefficients(np.squeeze(contour))[0])

# pyefd.plot_efd(efd_coeff, locus=dc_coeffs, image=img)

coeffs = efd_coeff.flatten()[3:] # type: ignore

print(coeffs)

# def efd_feature(contour):
#     coeffs = elliptic_fourier_descriptors(
#         contour, order=10, normalize=True)
#     return coeffs.flatten()[3:]
