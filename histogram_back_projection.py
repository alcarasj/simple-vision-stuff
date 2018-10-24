"""
https://docs.opencv.org/3.3.1/dc/df6/tutorial_py_histogram_backprojection.html

Histogram back-projection.

"""

import numpy
import cv2
import sys
from argparse import ArgumentParser

COLOR = cv2.IMREAD_COLOR
GRAYSCALE = cv2.IMREAD_GRAYSCALE
UNCHANGED = cv2.IMREAD_UNCHANGED

parser = ArgumentParser(description="Histogram back-projection.")
parser.add_argument("-i", "--input", dest="input_path",
					help="Input path for the image.")
args = parser.parse_args()
input_path = args.input_path

if not input_path:
	print("Specify an input path for the image as an argument.")
else:
	# Load the image.
	input_image = cv2.imread(input_path)

	# Select a region of interest from the image. 
	roi = cv2.selectROI(input_image, fromCenter=False)
	roi = input_image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

	# Convert BGR to HSV.
	roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	input_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

	# Calculate histogram.
	roi_hist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

	# Normalize histogram.
	cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

	# Apply back-projection.
	dst = cv2.calcBackProject([input_hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

	# Apply a convolution with circular disc.
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	cv2.filter2D(dst, -1, disc, dst)

	# Threshold and bitwise AND.
	ret, threshold = cv2.threshold(dst, 50, 255, 0)
	threshold = cv2.merge((threshold, threshold, threshold))
	result = cv2.bitwise_and(input_image, threshold)
	result = numpy.vstack((input_image, threshold, result))

	cv2.imshow("Result", result)
	cv2.waitKey(0)
	cv2.destroyAllWindows()