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
glasses_mask = np.zeros((rows, cols), np.uint8)
glasses = cv2.imread("glasses.jpeg")
r, c, _ = glasses.shape
for i in range(r):
	for j in range(c):
		x = glasses[i,j]
		flag=0
		for r in x:
			if r<=200:
				flag=1
				break
		if flag==0:
			glasses[i,j]=(0,0,0)
while True:
	_, frame = cap.read()		#read the frame
	glasses_mask.fill(0)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)	# convert into gray , easier to process
	
	
   	faces = detector(gray)		# apply detector

    	for face in faces:        
        	landmarks = predictor(gray, face)		# for each face in current frame landmarks = 68 points
		
		left = (landmarks.part(17).x ,landmarks.part(17).y)		# top glasses
		right = (landmarks.part(26).x ,landmarks.part(26).y)		# top glasses
		
		
		
		width = hypot( left[0] - right[0] , left[1]-right[1])			# width of glasses to find size of glasses image
														# hypot function from math to find distance
		height = width *0.3694						# to keep proportion of image constant (calc proportn fro  img)
		
		glasses1 = cv2.resize(glasses , (int(width),int(height)))			# resize image
		glasses_gray=cv2.cvtColor(glasses1,cv2.COLOR_BGR2GRAY)
		_,glasses_mask = cv2.threshold(glasses_gray ,1,255 , cv2.THRESH_BINARY_INV)	# mask to extract only glasses part , 
		# rectangle image required on pt 30 
		top_left = (int(left[0]),int(left[1]))
		
		#cv2.rectangle( frame ,(top_left[0],top_left[1]) ,(bottom_right[0],bottom_right[1]) ,(0,255,0) ,2)	
						
		glasses_area=frame[ top_left[1]: top_left[1]+int(height) , top_left[0] :top_left[0]+int(width)]
		glasses_area_no_glasses = cv2.bitwise_and(glasses_area, glasses_area, mask=glasses_mask)
		
		       
        	final_glasses = cv2.add(glasses_area_no_glasses, glasses1)
        	frame[top_left[1]: top_left[1] + int(height),top_left[0]: top_left[0] + int(width)] = final_glasses
           
            
        '''# output = face_utils.visualize_facial_landmarks(frame, landmarks)
    	cv2.imshow(" Frame",frame)
	cv2.imshow("glasses1" , glasses_mask)
	#cv2.imshow("glassesMask" , glasses_mask)'''
	try:		
    		cv2.imshow(" Frame",frame)
		cv2.imshow("glasses1" , glasses_area_no_glasses)
		
	except:
		#print("NO Face Detected")
		cv2.putText(frame,'NO Face Detected',(rows/2,cols/2), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
		cv2.imshow(" Frame",frame)

    	key = cv2.waitKey(1)

    	if key == 27 or key == ord('q'):			# 27=>esc
        	break
