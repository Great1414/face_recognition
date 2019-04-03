#!/usr/bin/env python
# encoding: utf-8
'''
@author: Ren
@file: face_match.py
@time: 2019-04-02 19:19
@desc: 人脸对比识别
'''
import dlib
import os
import cv2
import numpy as np
import pandas as pd

def cal_distance(feature1, feature2):
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    dist = np.sqrt(np.sum(np.square(feature1 - feature2)))
    if dist > 0.4:
        return "diff"
    else:
        return "same"


features_known_csv = "../result/result_features/"
features_rd = pd.read_csv(features_known_csv, header=None)
features_known_arr = []
for i in range(features_known_csv.shape[0]):
    features_someone_arr = []
    for j in range(0, len(features_known_csv.ix[i, :])):
        features_someone_arr.append(features_known_csv.ix[i,:][j])
    features_someone_arr.append(features_someone_arr)
print("faces in database:", len(features_known_arr))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../dlib_model/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("../dlib_model/dlib_face_recognition_resnet_model_v1.dat")

cap = cv2.VideoCapture(0)
cap.set(3, 480)
while cap.isOpened():
    flag, img_rd = cap.read()
    key = cv2.waitKey(1)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos_list = []
    name_list = []
    if key == ord("q"):
        break
    else:

        if len(faces) != 0:
            features_cap_arr = []
            for i in range(len(faces)):
                shape = predictor(img_rd, faces[i])
                features_cap_arr.append(facerec.compute_face_descriptor(img_rd,shape))

            for k in range(len(faces)):
                name_list.append("unknown")
                pos_list.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top())/4)]))

                for j in range(len(features_known_arr)):
                    compare = cal_distance(features_cap_arr[k], features_known_arr[j])
                    if compare == "same":
                        if j == 0:
                            name_list[k] == "person1"
                        else:
                            name_list[k] == "person2"

                for kk, d in enumerate(faces):
                    cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)

            for i in range(len(faces)):
                cv2.putText(img_rd, name_list[i], pos_list[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("camera",img_rd)

cap.release()
cv2.destroyALLWindows()