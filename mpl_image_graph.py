import cv2
import numpy
import matplotlib.image as mplimg
import matplotlib.pyplot as mplppl


# open image/video
media_path = "/home/arko/Mini_Project/test_images/urban_marked/um_000002.png"
image = mplimg.imread(media_path)

height = image.shape[0]
width = image.shape[1]

region_of_interest_vertices = [
    (180, height),
    (550, int(height/2)),
    (620, int(height/2)),
    (820, height),
]

def region_of_interest(img, vertices):
    mask = numpy.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# convert to grayscale
gray_cv2pic = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

cropped_image = region_of_interest(gray_cv2pic, numpy.array([region_of_interest_vertices], numpy.int32))

mplppl.figure()
mplppl.imshow(cropped_image)
mplppl.show()