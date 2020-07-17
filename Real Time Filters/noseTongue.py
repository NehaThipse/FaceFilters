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
tongue = cv2.imread("Images/tongue.jpeg")
nose = cv2.imread("Images/nose.png")

#-----------------------
r, c, _ = tongue.shape
for i in range(r):
	for j in range(c):
		x = tongue[i,j]
		flag=0
		for r in x:
			if r<=200:
				flag=1
				break
		if flag==0:
			tongue[i,j]=(0,0,0)
#-------------------
t_mask=np.zeros((rows, cols), np.uint8)
cntimg=0
while True:
	_, frame = cap.read()		#read the frame

	t_mask.fill(0)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)	# convert into gray , easier to process

	
	
   	faces = detector(gray)		# apply detector

    	for face in faces:        
        	landmarks = predictor(gray, face)		# for each face in current frame landmarks = 68 points		
		up= (landmarks.part(62).x ,landmarks.part(62).y)	
		down= (landmarks.part(66).x ,landmarks.part(66).y)	
		t_left =(landmarks.part(60).x ,landmarks.part(60).y)	
		t_right =(landmarks.part(64).x ,landmarks.part(64).y)	
		p60=(landmarks.part(60).x ,landmarks.part(60).y)			
		dist = hypot(up[0]-down[0] , up[1]-down[1] )# aa
		#print(dist)		#.. found that dist ranges from 2/3 ... 30+
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
		
		#=============== tongue
		if dist>=8:
			t_width = hypot(t_left[0] - t_right[0] , t_left[1]-t_right[1])*1.2#*1.5 is temp
			t_height =1.23*t_width
			t1=cv2.resize(tongue , (int(t_width) , int(t_height)))		
			tongue_gray=cv2.cvtColor(t1,cv2.COLOR_BGR2GRAY)
			_,tongue_mask = cv2.threshold(tongue_gray ,10 ,255 , cv2.THRESH_BINARY_INV)
			tongue_area=frame[ p60[1]: p60[1]+int(t_height) , p60[0] :p60[0]+int(t_width)]
			tongue_area_no_t= cv2.bitwise_and(tongue_area, tongue_area, mask=tongue_mask)
			final_t = cv2.add(tongue_area_no_t , t1)
			frame[ p60[1]: p60[1]+int(t_height) , p60[0] :p60[0]+int(t_width)] = final_t
           
        #cv2.imshow("TMask" , tongue_gray)   
    	#cv2.imshow("TA" , tongue_area)
    	cv2.imshow(" Frame",frame)
	cv2.imwrite(filename="Images/output/img"+str(cntimg)+".png" , img=frame)
	cntimg+=1
	#cv2.imshow("Nose1" , tongue_area_no_t)
	#cv2.imshow("NoseMask" , nose_mask)
	

    	key = cv2.waitKey(1)

    	if key == 27 or key == ord('q'):			# 27=>esc
        	break
