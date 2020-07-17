# FaceFilters using Facial Landmarks

This project in python can put various snapchat like filters such as nose , glasses ,mustache , tongue (on opening mouth) on detected face. 
Face is detected using dlib and filters are put on images(real time / input image) by identifying landmarks using 68 point landmark detector using opencv operations and masking.<br>

### Steps -
1. Detect faces real time using dlib library.
2. For each face get 68 landmark points using 68 point landmark detector.The file is available at http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
3. Get required co-ordinates (Ex: for nose co-ordinates associated with nose area)
4. Calculate width and height ratio from co-ordinates in order to resize filter image to be put on face.(If we come closer to camera , size of nose filter image should increase in proportion wih the real nose)
5. Apply masking to extract only nose/filter part from image and not the background.
6. Replace part in original image by this masked filter.

### output
![output image](Output/Output.jpg)
<img src="Output/Output.gif" width="200" height="200">

### Reference 
https://www.youtube.com/watch?v=IJpTe-1cimE<br>
This video is for pig's nose filter. I used similar steps to create other filters.

