import numpy
import cv2
import sys

camera = cv2.VideoCapture(0)

while(1):
	ret, frame = camera.read()
	face_cascade = cv2.CascadeClassifier("./env/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml")
	eye_cascade = cv2.CascadeClassifier("./env/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml")
	grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)
	for (x, y, width, height) in faces:
		cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
		roi_grayscale = grayscale[y:y + height, x:x + width]
		roi_color = frame[y:y + height, x:x + width]
		eyes = eye_cascade.detectMultiScale(roi_grayscale)
		for (eye_x, eye_y, eye_width, eye_height) in eyes:
			cv2.rectangle(roi_color, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height), (0, 255, 0), 2)

	cv2.imshow("Face Detection", frame)

	key = cv2.waitKey(5) & 0xFF
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()