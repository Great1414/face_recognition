#!/usr/bin/env python
# encoding: utf-8
'''
@author: Ren
@file: faces_from_camera.py
@time: 2019-04-01 19:15
@desc: 读取人脸
'''
import dlib
#import shutil
import cv2
import os
import numpy as np

# 创建文件夹
def create_work_file(path_photos,path_csv):
    if os.path.isdir(path_photos):
        pass
        print("exit")
    else:
        os.mkdir(path_photos)
    if os.path.isdir(path_csv):
        pass
    else:
        os.mkdir(path_csv)


# 人脸录入
def face_register(path_photos, cap, detector):
    if os.listdir(path_photos):
        person_list = os.listdir(path_photos).sort()
        person_end = int(str(person_list[-1]).split("_")[-1])
        person_sum = person_end

    else:
        person_sum = 0

    save_fig = 1
    press_n_before = 0

    while cap.isOpened():
        flag, img_rd = cap.read()
        key = cv2.waitKey(1)
        img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray,0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 新建，按键"n"
        if key == ord("n"):
            person_sum += 1
            current_face_path = path_photos + "person_" + str(person_sum)
            os.mkdirs(current_face_path)
            cnt_faces = 0
            press_n_before = 1

        if len(faces) != 0:
            for k, d in enumerate(faces):
                # 矩形框
                pos_start = tuple([d.left(), d.top()])
                pos_end = tuple([d.right(), d.bottom()])
                height = (d.bottom() - d.top())
                width = (d.right() - d.left())
                hh = int(height/2)
                ww = int(width/2)

                # 设置颜色
                color_rect = (255, 255, 255)
                if (d.right() + ww) >640 or (d.bottom() + hh >480) or \
                    (d.left() - ww < 0 ) or (d.top() - hh < 0 ):
                    cv2.putText(img_rd, "out of range", (20, 300), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    color_rect = (0, 0, 255)
                else:
                    color_rect = (255, 255, 255)
                    save_fig = 1
                cv2.rectangle(img_rd,
                              tuple([d.left() - ww, d.top() - hh]),
                              tuple([d.right() + ww, d.bottom() + hh]),
                              color_rect, 2)
                # 人脸大小空图像
                img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

                # save fig
                if save_fig:
                    # 按"s"
                    if key == ord("s"):
                        if press_n_before:
                            cnt_faces += 1
                            for h in range(height*2):
                                for w in range(width*2):
                                    img_blank[h][w] = img_rd[d.top() - hh + h][d.left() - ww + w]
                            cv2.imwrite(current_face_path + "/img_face_" + str(cnt_faces) + ".jpg", img_blank)
                            print("写入本地into:", str(current_face_path + "/img_face_" + str(cnt_faces) + ".jpg"))
                        else:
                            print("请在按's'之前，先按'n'")

        # 显示人脸数
        cv2.putText(img_rd, "faces:" + str(len(faces)), (20,100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        # 添加说明
        cv2.putText(img_rd, "Face Register", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "N: New face folder", (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "S: Save current face", (20, 400), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        # 按下'q'退出
        if key == "q":
            break

        cv2.imshow("camera", img_rd)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # dlib 正面人脸检测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./dlib_model/shape_predictor_68_face_landmarks.dat")
    # opencv调用摄像头
    cap = cv2.VideoCapture(0)
    cap.set(3, 480)
    print(cap)
    path_photos = "../data/data_faces_from_camera/"
    path_csv = "../data/data_csvs_from_photos/"
    create_work_file(path_photos, path_csv)
    face_register(path_photos, cap, detector)
















                











