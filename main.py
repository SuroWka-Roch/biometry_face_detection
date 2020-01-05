#!/usr/bin/env python3.7
import cv2 as cv
import numpy as np
import face_tools as ft
import matplotlib.pyplot as plot

from glob import glob as glob

OUTPUT_FOLDER = "./output/"


def use_a_darn_matplot_to_get_histograme(histograme, filename):
    plot.figure()
    plot.barh([x for x in range(len(histograme))], histograme)
    plot.savefig(filename)


file_names = glob("./fig/*")

for num, file_name in enumerate(file_names):
    pic = cv.imread(file_name)
    cv.imshow("oryginal", pic)
    cv.imwrite(OUTPUT_FOLDER + "org_" + str(num) + ".jpg", pic)

    pic_rbg = ft.face_binary_RBG(pic)
    pic_hsv = ft.face_binary_HSV(pic)
    pic_both = ft.pic_AND(pic_hsv, pic_rbg)

    pic_both = ft.erode_dilate_for(10, pic_both)
    cv.imshow("face_detection", pic_both)
    cv.imwrite(OUTPUT_FOLDER + "face" + str(num) + ".jpg", pic_both)

    ft.check_proportions(pic_both)
    pic_mask = ft.fill_holes(pic_both)
    cv.imshow("mask", pic_mask)
    cv.imwrite(OUTPUT_FOLDER + "mask" + str(num) + ".jpg", pic_mask)

    pic_seperated = ft.apply_mask(pic, pic_mask)
    cv.imshow("maskrbg", pic_seperated)
    cv.imwrite(OUTPUT_FOLDER + "mrbg" + str(num) + ".jpg", pic_seperated)

    pic_seperated = ft.separate_elements(pic_seperated, pic_mask)
    use_a_darn_matplot_to_get_histograme(
        ft.get_vertical_histogram(pic_seperated),
        OUTPUT_FOLDER + "hist" + str(num) + ".png")

    cv.imshow("final", pic_seperated)
    cv.imwrite(OUTPUT_FOLDER + "finl" + str(num) + ".jpg", pic_seperated)

    cv.waitKey()
