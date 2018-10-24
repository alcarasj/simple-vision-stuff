"""
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html

Real-time (webcam) face detection using Haar cascades.

"""

import numpy
import cv2
import sys

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("./assets/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./assets/haarcascade_eye.xml")

while True:
	# Keep reading frames from the camera.
	(grabbed, frame) = camera.read()
	grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Find the faces in the image.
	faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)
	for (x, y, width, height) in faces:
		# Draw blue rectangles around the detected faces.
		cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

		# Use the faces as regions of interest to detect the eyes.
		roi_grayscale = grayscale[y:y + height, x:x + width]
		roi_color = frame[y:y + height, x:x + width]

		# Find the eyes within the face using the region of interest.
		eyes = eye_cascade.detectMultiScale(roi_grayscale)
		for (eye_x, eye_y, eye_width, eye_height) in eyes:
			# Draw green rectangles around the detected eyes.
			cv2.rectangle(roi_color, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height), (0, 255, 0), 2)

	# Show the frame with the detected faces/eyes.
	cv2.imshow("Face Detection", frame)

	# Press ESC key to quit.
	key = cv2.waitKey(5) & 0xFF
	if key == 27:
		break

camera.release()
cv2.destroyAllWindows()