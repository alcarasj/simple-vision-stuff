import numpy
import cv2
import sys

LOWER_BOUND = numpy.array([0, 48, 80], dtype="uint8")
UPPER_BOUND = numpy.array([20, 255, 255], dtype="uint8")

camera = cv2.VideoCapture(0)

while(1):
	(grabbed, frame) = camera.read()
	converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	skin_mask = cv2.inRange(converted, LOWER_BOUND, UPPER_BOUND)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
	skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

	skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
	skin = cv2.bitwise_and(frame, frame, mask=skin_mask)

	cv2.imshow("Skin Detection", skin)

	key = cv2.waitKey(5) & 0xFF
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()