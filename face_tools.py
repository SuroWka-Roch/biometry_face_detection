def face_binary_RBG(pic):
    pic = pic.astype(int)
    pic = pic[:,:,2]-pic[:,:,1]
    pic[pic < 0] = 0
    pic = pic.astype( np.uint8 )
    t,bin_rgb =cv.threshold(pic,np.average(pic),255 ,cv.THRESH_OTSU)
    return bin_rgb


def face_binary_HSV(pic):
    pic_hsv = cv.cvtColor(pic, cv.COLOR_BGR2HSV)
    
    v1 = pic_hsv[:,:,0]<50
    v2 = pic_hsv[:,:,1] < 0.69*255
    v2_2 = pic_hsv[:,:,1] > 0.22*255
    v3 = pic_hsv[:,:,2] > 0.4*255

    v1 = np.logical_and(v1,v2)
    v1 = np.logical_and(v1,v2_2)
    v1 = np.logical_and(v1,v3)

    hsv_binary = np.zeros((pic_hsv.shape[0],pic_hsv.shape[1],3), np.uint8)
    hsv_binary[v1] = np.array([255,255,255])
    return pic_hsv

if __name__ == "__main__":
    pass