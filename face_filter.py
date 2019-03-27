#Importings
import cv2
import math
import numpy as np
import dlib

#Video Caputring and facial landmarks detector initialization
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
	#Reading camera and grayscaling
	ret, frame = cap.read()
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#Face and landmark detection
	faces = detector(gray_frame)
	landmarks = predictor(gray_frame, faces[0])

	#Nose width calculation based on the nose landmarks
	# 31 and 35 for horizontal
	nose_width = int(math.sqrt(pow(landmarks.part(35).y - landmarks.part(31).y, 2) + pow(landmarks.part(35).x - landmarks.part(31).x, 2))) + 22
	# 28 and 31 for vertical		
	#nose_height = int(math.sqrt(pow(landmarks.part(31).y - landmarks.part(28).y, 2) + pow(landmarks.part(31).x - landmarks.part(28).x, 2)))
	
	#Importing and thresholding wrt to the background
	nose_img = cv2.imread('/home/vatsalbabel/Desktop/Face Filters/pig_nose.png')
	nose_img = cv2.resize(nose_img, (nose_width, nose_width))
	gray_nose_img = cv2.cvtColor(nose_img, cv2.COLOR_BGR2GRAY)
	_, mask = cv2.threshold(gray_nose_img, 25, 255, cv2.THRESH_BINARY)
	nose_area = frame[int(landmarks.part(30).y - nose_width/2) : int(landmarks.part(30).y - nose_width/2) + nose_width, int(landmarks.part(30).x - nose_width/2) : int(landmarks.part(30).x - nose_width/2) + nose_width]
	nose_without_black = cv2.bitwise_and(nose_img, nose_img, mask = mask)
	_, mask = cv2.threshold(gray_nose_img, 25, 255, cv2.THRESH_BINARY_INV)
	re_nose_area = cv2.bitwise_and(nose_area, nose_area, mask = mask)
	nose_img = cv2.add(nose_without_black, re_nose_area)

	#Overlapping over the original frame
	frame[int(landmarks.part(30).y - nose_width/2) : int(landmarks.part(30).y - nose_width/2) + nose_width, int(landmarks.part(30).x - nose_width/2) : int(landmarks.part(30).x - nose_width/2) + nose_width] = nose_img
	
	#Displaying
	cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF==ord('q'):
		break

#Releasing camera
cv2.destroyAllWindows()
cap.release()
