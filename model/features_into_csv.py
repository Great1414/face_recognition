#!/usr/bin/env python
# encoding: utf-8
'''
@author: Ren
@file: features_into_csv.py
@time: 2019-04-02 15:51
@desc: 提取人脸特征
'''
import cv2
import os
import dlib
import csv
from skimage import io
import pandas as pd
import numpy as np

# 提取128D特征
def get_128D_features(img_path, detector, predictor, facerec):
    img = io.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray, 1)
    if len(faces) != 0:
        shape = predictor(img_gray, faces[0])
        face_descriptor = facerec.computer_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("no find faces")
    return face_descriptor

# 特征写入csv
def write_into_csv(path_faces_personx, path_csv_from_photos, detector, predictor, facerec):
    photos_list = os.listdir(path_faces_personx)
    with open(path_csv_from_photos, "w", newline= "") as csvfile:
        writer = csv.writer(csvfile)
        if photos_list:
            for each in range(len(photos_list)):
                print("正在读取图像", path_faces_personx + "/" + photos_list[each])
                features_128D = get_128D_features(path_faces_personx + "/" + photos_list[each], detector, predictor, facerec)
                if features_128D == 0:
                    each += 1
                else:
                    writer.writerow(features_128D)
        else:
            print("文件夹内图像为空")
            writer.writerow("")

# 某个personx的所有图像的特征，写入personx.csv
def get_personx_features(path_photos_from_camera, path_csvs_from_photos,detector, predictor, facerec):
    faces = os.listdir(path_photos_from_camera)
    faces.sort()
    for person in faces:
        print(path_csvs_from_photos + person + ".csv")
        write_into_csv(path_photos_from_camera + person, path_csvs_from_photos,detector, predictor, facerec)

# 从CSV中读取数据，计算128D均值
def average_128D(path_csvs_from_photos):
    column_names = []
    for feature in range(128):
        column_names.append("feature_" + str(feature + 1))

    df = pd.read_csv(path_csvs_from_photos, names= column_names)

    if df.size != 0:
        feature_mean_list = []
        for feat in range(128):
            tmp = df["feature_" + str(feat + 1)]
            tmp = np.array(tmp)
            tmp_mean = np.mean(tmp)
            feature_mean_list.append(tmp_mean)
    else:
        feature_mean_list = []
    return feature_mean_list

if __name__ == "__main__":
    path_photos_from_camera = "../data/data_faces_from_camera/"
    path_csvs_from_photos = "../data/data_csvs_from_photos/"
    path_csv_from_photos_features = "../result/result_features/"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./dlib_model/shape_predictor_5_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("./dlib_model/dlib_face_recognition_resnet_model_v1.dat")
    get_personx_features(path_photos_from_camera, path_csvs_from_photos, detector, predictor, facerec)
    with open(path_csv_from_photos_features, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        csv_rd = os.listdir(path_csvs_from_photos)
        csv_rd.sort()
        print("feature均值")
        for i in range(len(csv_rd)):

            feature_mean_list = average_128D(path_csvs_from_photos + csv_rd[i])
            print(path_csvs_from_photos + csv_rd[i])
            writer.writerow(feature_mean_list)








