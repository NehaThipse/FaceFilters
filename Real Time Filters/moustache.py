import cv2
import numpy as np 
import dlib
from imutils import face_utils
from math import hypot
import numpy as np

cap = cv2.VideoCapture(0)		# 0->index of camera
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
_, frame = cap.read()
rows, cols, _ = frame.shape
nose_mask = np.zeros((rows, cols), np.uint8)

while True:
	_, frame = cap.read()		#read the frame
	nose_mask.fill(0)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)	# convert into gray , easier to process
	m = cv2.imread("m.png")
	
   	faces = detector(gray)		# apply detector

    	for face in faces:        
        	landmarks = predictor(gray, face)		# for each face in current frame landmarks = 68 points
		up = (landmarks.part(33).x ,landmarks.part(33).y)		# nose
		down = (landmarks.part(50).x ,landmarks.part(50).y)		# upper lip		
		height = hypot( up[0] - down[0] , up[1]-down[1])			# width of nose to find size of nose image
														# hypot function from math to find distance
		width = height *4							# to keep proportion of image constant (calc proportn fro  img)
		
		m1 = cv2.resize(m , (int(width),int(height)))			# resize image
		m_gray=cv2.cvtColor(m1,cv2.COLOR_BGR2GRAY)
		_,m_mask = cv2.threshold(m_gray ,10 ,255 , cv2.THRESH_BINARY)
		# mask to extract only nose part , 0-> threshold value ( trial)
		# rectangle image required on pt 30 
		top_left = (int(up[0]-width/2),int(up[1]))
		#bottom_right = (int(center[0]+width/2),int(center[1]+height/2))
		#cv2.rectangle( frame ,(top_left[0],top_left[1]) ,(bottom_right[0],bottom_right[1]) ,(0,255,0) ,2)	
						
		m_area=frame[ top_left[1]: top_left[1]+int(height) , top_left[0] :top_left[0]+int(width)]
		m_area_no_m = cv2.bitwise_and(m_area, m_area, mask=m_mask)
		
		       
        	#final_nose = cv2.add(nose_area_no_nose, nose1)
        	frame[top_left[1]: top_left[1] + int(height),top_left[0]: top_left[0] + int(width)] = m_area_no_m
		
		

	

           
            
    # output = face_utils.visualize_facial_landmarks(frame, landmarks)
    	cv2.imshow(" Frame",frame)
	#cv2.imshow("Nose1" , nose_area_no_nose)
	#cv2.imshow("NoseMask" , nose_mask)
	

    	key = cv2.waitKey(1)

    	if key == 27 or key == ord('q'):			# 27=>esc
        	break
