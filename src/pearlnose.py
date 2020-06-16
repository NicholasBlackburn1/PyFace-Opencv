"""
Python Opencv Face Repacer / move  3d model useing onlyface
"""

import cv2
import numpy as np
import dlib
from math import hypot
import matplotlib.pyplot as plt
import threading
from matplotlib import animation

cap = cv2.VideoCapture(0)



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
nose_image = cv2.imread("1.png")
eye_image = cv2.imread("eyeL.png")

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (44,426)
bottomLeftCornerOfText2 = (44,464)
bottomLeftCornerOfText3 = (44,440)
fontScale              = .5
fontColor              = (255,255,255)
lineType               = 2


def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


   


       
    
    
    
    
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows, cols, _ = frame.shape
    nose_mask = np.zeros((rows, cols), np.uint8)
    eyeLmask = np.zeros((rows,cols), np.uint8)
    nose_mask.fill(0)
    eyeLmask.fill(0)
   
    faces = detector(gray)
    for face in faces:
        face_left = face.left()
        face_top = face.top()
        face_right = face.right()
        face_bottom = face.bottom()
        cv2.rectangle(frame, (face_left, face_top), (face_right, face_bottom), (0, 255, 0), 3)
        
        

        landmarks = predictor(gray, face)
        '''
        face landmark display and math
        '''
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            
            nose_top = (landmarks.part(29).x, landmarks.part(29).y)
            nose_center = (landmarks.part(30).x, landmarks.part(30).y)
            nose_left = (landmarks.part(31).x, landmarks.part(31).y)
            nose_right = (landmarks.part(35).x, landmarks.part(35).y)

            
            right_eye_left_point = (landmarks.part(42).x, landmarks.part(42).y)
            right_eye_right_point = (landmarks.part(45).x, landmarks.part(45).y)
            
            left_eye_left_point = (landmarks.part(36).x, landmarks.part(36).y)
            left_eye_right_point = (landmarks.part(39).x, landmarks.part(39).y)
            
            left_eye_center_top = midpoint(landmarks.part(37), landmarks.part(38))
            left_eye_center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

            right_eye_center_top= midpoint(landmarks.part(43), landmarks.part(44))
            right_eye_center_bottom = midpoint(landmarks.part(46), landmarks.part(47))

            left_eye_hor_line = cv2.line(frame, left_eye_left_point, left_eye_right_point, (0, 255, 0), 2)
            left_eye_ver_line = cv2.line(frame, left_eye_center_top, left_eye_center_bottom, (0, 255, 0), 2)
            
            right_eye_hor_line = cv2.line(frame, right_eye_left_point, right_eye_right_point, (0, 255, 0), 2)
            right_eye_ver_line = cv2.line(frame, right_eye_center_top, right_eye_center_bottom, (0, 255, 0), 2)
            
           
            eyeL_width = (hypot(left_eye_left_point[0] - left_eye_right_point[0], left_eye_left_point[1] - left_eye_right_point[1]))
            eyeL_center = (hypot(left_eye_center_top[0] -left_eye_center_bottom[0],left_eye_center_top[1] - left_eye_center_bottom[1]))
            eyeL_height = (eyeL_width*0.77)
            
            nose_width = (hypot(nose_left[0] - nose_right[0], nose_left[1]- nose_right[1] ))
            nose_height = (nose_width*0.77)
            
            
            
            
                    # New nose position
            top_left = (int(nose_center[0] - nose_width / 2),
                              int(nose_center[1] - nose_height / 2))
            
            bottom_right = (int(nose_center[0] + nose_width / 2),
                       int(nose_center[1] + nose_height / 2))
            
            top_left_eye = (int(eyeL_center[0] - eyeL_width / 2),
                              int(eyeL_center[1] - eyeL_height / 2))
            
            bottom_right_eye = (int(eyeL_center[0] + eyeL_width / 2),
                int(eyeL_center[1] + eyeL_height / 2))
            
            dim =(int(nose_width),int(nose_height))
            Lefteyedim =(int(eyeL_width),int(eyeL_height))
            
            cv2.putText(frame,'nose pos new'+str(dim), 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            
            cv2.putText(frame,'eyeLEFT pos new'+str(Lefteyedim), 
                bottomLeftCornerOfText2, 
                font, 
                fontScale,
                fontColor,
                lineType)
            
           
            cv2.putText(frame,'eyeLEFT centerpoint'+str(int(top_left_eye),int(bottom_right_eye)), 
                bottomLeftCornerOfText3, 
                font, 
                fontScale,
                fontColor,
                lineType)
            
            
            
            nose_pig = cv2.resize(nose_image, dim)
            nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
            _,  nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)   
           
           
            nose_area = frame[int(top_left[1]): int(top_left[1]) + int(nose_height),
            int(top_left[0]): int(top_left[0]) + int(nose_width)]
            
            nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
            final_nose = cv2.add(nose_area_no_nose, nose_pig)
            
            frame[int(top_left[1]): int(top_left[1]) + int(nose_height),
            int(top_left[0]): int(top_left[0]) + int(nose_width)] = final_nose
            
            #Eye area and masking 
            
            lefteye = cv2.resize(eye_image, Lefteyedim)
            left_eye_gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
            _,  eye_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)   
           
           
            Left_eye_area = frame[int(top_left_eye[1]): int(top_left_eye[1]) + int(eyeL_height),
            int(top_left_eye[0]): int(top_left_eye[0]) + int(eyeL_width)]

            eyeL_area_no_eye = cv2.bitwise_and(Left_eye_area, Left_eye_area, mask=eye_mask)
            final_Left_eye = cv2.add(eyeL_area_no_eye, eye_image)
           
           
            #cv2.imshow("Nose pig", eye_image)
            cv2.imshow("final nose", final_nose)
            #cv2.imshow("finalEyee",gray)
            
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
   
        
    cv2.imshow("Frame", frame)
    #cv2.imshow("gray", gray)

        

 
    #cv2.imshow("Nose pig", nose_image)

    key = cv2.waitKey(1)
    if key == 27:
        break