import numpy
import cv2
import sys

COLOR = cv2.IMREAD_COLOR
GRAYSCALE = cv2.IMREAD_GRAYSCALE
UNCHANGED = cv2.IMREAD_UNCHANGED

if len(sys.argv) < 2:
	print("Specify a path to an input image as an argument.")
else:
	image_path = sys.argv[1]
	image = cv2.imread(image_path, GRAYSCALE)
	cv2.imshow("image", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
