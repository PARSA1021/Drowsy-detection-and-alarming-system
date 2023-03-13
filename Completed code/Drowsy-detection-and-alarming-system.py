#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import dlib
from scipy.spatial import distance
import winsound
import time

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    EAR = (A+B)/(2.0*C)
    return EAR

BLINK_THERSHOLD = 0.13
CONSEC_FRAMES_NUMBER = 2

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Declare another costant to hold the consecutive number of frames to consider for a blink
CONSECUTIVE_FRAMES = 20
# Initialize two counters
TOTAL = 0
COUNTER = 0
start_time = 0
ALARM_ON = False

while True:
    # Extract a frame
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        left_ear = eye_aspect_ratio(leftEye)
        right_ear = eye_aspect_ratio(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)

        print(EAR)

        if EAR < BLINK_THERSHOLD:
            COUNTER += 1
            if start_time == 0:
                start_time = time.time()
            else:
                end_time = time.time()
                if end_time - start_time > 2: ALARM_ON = True

                winsound.PlaySound("part3.wav", winsound.SND_FILENAME)

        else:
            if COUNTER >= CONSEC_FRAMES_NUMBER:
                TOTAL += 1
            COUNTER = 0
            start_time = 0
            ALARM_ON = False



        cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Blink: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if ALARM_ON:
            cv2.putText(frame, "ALERT!", (150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("EAR frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

