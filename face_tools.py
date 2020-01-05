import cv2 as cv
import numpy as np


def face_binary_RBG(pic):
    pic = pic.astype(int)
    pic = pic[:, :, 2]-pic[:, :, 1]
    pic[pic < 0] = 0
    pic = pic.astype(np.uint8)
    t, bin_rgb = cv.threshold(pic, np.average(pic), 255, cv.THRESH_OTSU)
    return bin_rgb


def face_binary_HSV(pic):
    pic_hsv = cv.cvtColor(pic, cv.COLOR_BGR2HSV)

    v1 = pic_hsv[:, :, 0] < 50
    v2 = pic_hsv[:, :, 1] < 0.69*255
    v2_2 = pic_hsv[:, :, 1] > 0.22*255
    v3 = pic_hsv[:, :, 2] > 0.4*255

    v1 = np.logical_and(v1, v2)
    v1 = np.logical_and(v1, v2_2)
    v1 = np.logical_and(v1, v3)

    hsv_binary = np.zeros((pic_hsv.shape[0], pic_hsv.shape[1]), np.uint8)
    hsv_binary[v1] = 255
    return hsv_binary


def pic_AND(pic1, pic2):
    and_table = np.logical_and(pic1 == 255, pic2 == 255)
    and_pic = np.zeros((pic1.shape[0], pic1.shape[1]), np.uint8)
    and_pic[and_table] = 255
    return and_pic

def erode_dilate_for(number, pic):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    for _ in range(number):
        pic = cv.erode(pic, kernel)
        pic = cv.dilate(pic, kernel)

    return pic

def check_proportions(pic,tolerance=0.6):
    """Take binary return only spaces with golden proportions"""
    golden_proportion = 1.6
    im = pic.copy()
    
    picRBG = cv.cvtColor(im, cv.COLOR_GRAY2BGR)

    # Find the contour of the leming
    contour,_ = cv.findContours(pic.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)


    # And draw it on the original image
    for c in contour:
        # enter your filtering here
        x,y,w,h = cv.boundingRect(c)
        cv.rectangle(picRBG,(x,y),(x+w,y+h),(0,255,0),2)
        #find proportion
        proportion = h/w
        if abs(proportion-golden_proportion)>tolerance:
            pic = cv.fillConvexPoly(pic,c,0)

    cv.imshow("removed non perfect",pic)
    cv.waitKey(0)

def fill_holes(img):
    """Fill holes in binary image"""
    hight = img.shape[0]
    width = img.shape[1]

    im_flood = ~img.copy()

    mask = np.zeros((hight+2, width+2), np.uint8)
    cv.floodFill(im_flood, mask, (0, 0), 255)

    pic_final = im_flood & img

    return pic_final



if __name__ == "__main__":
    pass
