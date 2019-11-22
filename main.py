#!/usr/bin/env python3.7
import cv2 as cv
import numpy as np
import face_tools as ft

from glob import glob as glob


file_names = glob("./fig/*")

for file_name in file_names:
    #GRB
    pic = cv.imread(file_name)
    pic_original = pic
    pic = pic.astype(int)
    pic = pic[:,:,2]-pic[:,:,1]
    pic[pic < 0] = 0
    pic = pic.astype( np.uint8 )
    t,bin_rgb =cv.threshold(pic,np.average(pic),255 ,cv.THRESH_OTSU)

    #HSV
    pic_hsv = cv.cvtColor(pic_original, cv.COLOR_BGR2HSV)

    v1 = pic_hsv[:,:,0]<50
    v2 = pic_hsv[:,:,1] < 0.69*255
    v2_2 = pic_hsv[:,:,1] > 0.22*255
    v3 = pic_hsv[:,:,2] > 0.4*255

    v1 = np.logical_and(v1,v2)
    v1 = np.logical_and(v1,v2_2)
    v1 = np.logical_and(v1,v3)

    hsv_binary = np.zeros((pic_hsv.shape[0],pic_hsv.shape[1],3), np.uint8)
    hsv_binary[v1] = np.array([255,255,255])
    cv.imshow("name",hsv_binary)
    cv.waitKey()
