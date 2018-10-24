"""
https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/

Real-time (webcam) skin detection using masks.

"""

import numpy
import cv2
import sys

camera = cv2.VideoCapture(0)

# Define the range of skin color in HSV.
lower_bound = numpy.array([0, 48, 80], dtype="uint8")
upper_bound = numpy.array([20, 255, 255], dtype="uint8")

while True:
	# Keep reading frames from the camera.
	(grabbed, frame) = camera.read()

	# Convert BGR to HSV.
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Create a mask for skin isolation.
	mask = cv2.inRange(hsv, lower_bound, upper_bound)

	# Apply a series of erosions and dilations to the mask using an elliptical kernel.
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	mask = cv2.erode(mask, kernel, iterations=2)
	mask = cv2.dilate(mask, kernel, iterations=2)

	# Use a Gaussian blur the mask to help remove noise, then apply the mask to the frame.
	mask = cv2.GaussianBlur(mask, (3, 3), 0)
	skin = cv2.bitwise_and(frame, frame, mask=mask)

	# Show the detected skin in the frame.
	cv2.imshow("Skin Detection", skin)

	# Press ESC key to quit.
	key = cv2.waitKey(5) & 0xFF
	if key == 27:
		break

camera.release()
cv2.destroyAllWindows()