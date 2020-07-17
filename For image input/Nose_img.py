import cv2
import numpy as np 
import dlib
from imutils import face_utils
from math import hypot
import numpy as np


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
nose = cv2.imread("Images/nose.png") 
frame =cv2.imread("Images/img.jpg") 
rows, cols, _ = frame.shape
nose_mask = np.zeros((rows, cols), np.uint8)
nose_mask.fill(0)
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)	# convert into gray , easier to process
	
faces = detector(gray)		# apply detector

for face in faces:        
        landmarks = predictor(gray, face)		# for each face in current frame landmarks = 68 points
	top = (landmarks.part(29).x ,landmarks.part(29).y)		# top nose
	left = (landmarks.part(31).x ,landmarks.part(31).y)		# top nose
	right = (landmarks.part(35).x ,landmarks.part(35).y)		# top nose
	center= (landmarks.part(30).x ,landmarks.part(30).y)		# center nose
		
		
	width = hypot( left[0] - right[0] , left[1]-right[1])*1.8			# width of nose to find size of nose image
														# hypot function from math to find distance
	height = width *0.58								# to keep proportion of image constant (calc proportn fro  img)
		
	nose1 = cv2.resize(nose , (int(width),int(height)))			# resize image
	nose_gray=cv2.cvtColor(nose1,cv2.COLOR_BGR2GRAY)
	_,nose_mask = cv2.threshold(nose_gray ,10 ,255 , cv2.THRESH_BINARY_INV)	# mask to extract only nose part , 0-> threshold value ( trial)
		# rectangle image required on pt 30 
	top_left = (int(center[0]-width/2),int(center[1]-height/2))
	bottom_right = (int(center[0]+width/2),int(center[1]+height/2))
		#cv2.rectangle( frame ,(top_left[0],top_left[1]) ,(bottom_right[0],bottom_right[1]) ,(0,255,0) ,2)	
						
	nose_area=frame[ top_left[1]: top_left[1]+int(height) , top_left[0] :top_left[0]+int(width)]
	nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
		
		       
        final_nose = cv2.add(nose_area_no_nose, nose1)
        frame[top_left[1]: top_left[1] + int(height),top_left[0]: top_left[0] + int(width)] = final_nose
# output = face_utils.visualize_facial_landmarks(frame, landmarks)

try:	
	cv2.imshow("Nose1" , nose_area_no_nose)
	cv2.imshow(" Frame",frame)
	cv2.imwrite(filename='frame.jpg',img=frame)
except:
	print("NO Face Detected")
	cv2.putText(frame,'NO Face Detected',(rows/2,cols/2), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
	cv2.imshow(" Frame",frame)
	
	#cv2.imshow("NoseMask" , nose_mask)
	

key = cv2.waitKey(4000)


