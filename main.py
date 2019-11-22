#!/usr/bin/env python3.7
import cv2 as cv
import numpy as np
import face_tools as ft

from glob import glob as glob


file_names = glob("./fig/*")

for file_name in file_names:
    pic = cv.imread(file_name)
    pic_rbg = ft.face_binary_RBG(pic)
    pic_hsv = ft.face_binary_HSV(pic)
    pic_both = ft.pic_AND(pic_hsv, pic_rbg)
    cv.imshow("name1", pic_both)


    pic_both = ft.erode_dilate_for(10,pic_both)
    cv.imshow("name2", pic_both)
    cv.waitKey()
