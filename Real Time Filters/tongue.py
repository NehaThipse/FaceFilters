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
	#cv2.imshow("Nose1" , tongue_area_no_t)
	#cv2.imshow("NoseMask" , nose_mask)
	

    	key = cv2.waitKey(1)

    	if key == 27 or key == ord('q'):			# 27=>esc
        	break
