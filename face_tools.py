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

def check_proportions(pic):
    """Take binary return only spaces with golden proportions"""
    golden_proportion = 1.6
    im = pic.copy()
    
    #Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()

    params.blobColor= 255
    # params.filterByColor = True

    params.minThreshold = 0
    params.maxThreshold = 255

    params.minArea = 1500

    detector = cv.SimpleBlobDetector_create(params)
    

    # Detect blobs.
    keypoints = detector.detect(im)

    print(keypoints)
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Show keypoints
    cv.imshow("Keypoints", im_with_keypoints)
    cv.waitKey(0)



if __name__ == "__main__":
    pass
