"""
https://www.geeksforgeeks.org/real-time-edge-detection-using-opencv-python/

Real-time (webcam) Canny edge detection.

"""

import numpy
import cv2
import sys

camera = cv2.VideoCapture(0)

# Define the range of red color in HSV.
lower_red = numpy.array([30, 150, 50])
upper_red = numpy.array([255, 255, 180])

while True:
	# Keep reading frames from the camera.
	(grabbed, frame) = camera.read()

	# Convert BGR to HSV.
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Create a mask for edge isolation.
	mask = cv2.inRange(hsv, lower_red, upper_red)

	# Bitwise AND current frame and mask.
	res = cv2.bitwise_and(frame, frame, mask=mask)

	# Display the original frame.
	cv2.imshow("Original", frame)

	# Finds edges in the original frame and maps them to another frame as output. 
	edges = cv2.Canny(frame, 100, 200)
	cv2.imshow("Edges", edges)

	# Press ESC key to quit.
	key = cv2.waitKey(5) & 0xFF
	if key == 27:
		break

camera.release()
cv2.destroyAllWindows()