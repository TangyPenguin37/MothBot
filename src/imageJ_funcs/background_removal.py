from rembg import remove
import cv2

TEST_IMAGE = 'data/test_images/test2.jpg'

def show_image(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_bg(image):
    inputImage = cv2.imread(image)
    outputImage = remove(inputImage)

    return outputImage

show_image('test', cv2.imread(TEST_IMAGE))
show_image('test', remove_bg(TEST_IMAGE))