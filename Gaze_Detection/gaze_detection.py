from operator import rshift
import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import random
import pandas as pd
import xlsxwriter
import os
import glob 
left = 0
right = 0
center = 0
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ] #left eye area
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  #right eye area
RIGHT_IRIS = [474,475, 476, 477] #right iris area
LEFT_IRIS = [469, 470, 471, 472] #left iris area
L_H_LEFT = [33] #right eye right most landmark
L_H_RIGHT = [133] #right eye left most landmark
R_H_LEFT = [362] #left eye right most landmark
R_H_RIGHT = [263] #left eye left most landmark

#irisin konumunu bulabilmemiz icin gereken iki nokta arasındaki uzaklık fonksiyonu
def euclidean_distance(point1,point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2+(y2-y1)**2)
    return distance

#this function provide to find location of iris.
def iris_position(iris_center, right_point, left_point):
    center_to_right_distance = euclidean_distance(iris_center,right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_distance/total_distance
    iris_position =""
    if ratio > 0.42 and ratio <= 0.57:
        iris_position="1"
    elif ratio is None:
        iris_position="0"
    else:
        iris_position="0"
    return iris_position, ratio



# All files and directories ending with .txt and that don't begin with a dot:
videos = glob.glob("C:/Users/HP/OneDrive/Desktop/462_Gazedetection_Group2/videos/focus/*.mp4") #videoların nerede bulundugu
countIndex = 0
result=pd.DataFrame(columns=["Video_Name","Values"])
test = []

for i in videos:
    array = [] #we returned empty array cause we want to refill array in every video
    cap = cv.VideoCapture(i) #We open each video in the file one by one with the help of the for loop.
    def getFrame(sec):
        cap.set(cv.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = cap.read()
        #if machine can read the video, we can take a frame
        if hasFrames:
            os.chdir("C:/Users/HP/OneDrive/Desktop/462_Gazedetection_Group2/Frame/focus")
            cv.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
        return hasFrames
    print(i)
    os.chdir("../")
    os.getcwd()

   
    sec = 0
    frameRate=1  #it will capture image in each 1 second
    count=1
    success = getFrame(sec)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 1)
            success = getFrame(sec)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame,1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                cv.circle(frame, center_left, int(l_radius), (1,255,0), 1, cv.LINE_AA)
                cv.circle(frame, center_right, int(r_radius), (1,255,0), 1, cv.LINE_AA)
                cv.circle(frame, mesh_points[R_H_RIGHT][0], 1, (255,255,255), -1, cv.LINE_AA)
                cv.circle(frame, mesh_points[R_H_LEFT][0], 1, (0,255,255), -1, cv.LINE_AA)
                cv.circle(frame, mesh_points[L_H_RIGHT][0], 1, (255,255,255), -1, cv.LINE_AA)
                cv.circle(frame, mesh_points[L_H_LEFT][0], 1, (0,255,255), -1, cv.LINE_AA)
                iris_pos, ratio = iris_position(
                    center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0]
                )
                print(iris_pos)
                array.append(iris_pos)
                
                print(array)
                countIndex = countIndex+1
                cv.imshow('img', frame)
                key = cv.waitKey(1)
                if key ==ord('q'):
                    break
    test.append(array)
    result.loc[len(result)] = [i.replace('C:/Users/HP/OneDrive/Desktop/462_Gazedetection_Group2/videos/focus',''),array]
            
cap.release()
cv.destroyAllWindows()


os.chdir("../../") 
with pd.ExcelWriter("FocusResultFile.xlsx") as writer: 
      result.to_excel(writer, sheet_name='Result') 