import cv2
import numpy as np
import matplotlib.pyplot as plt
#thresholding is only applied on BW images as we need only the features.
#a very simple method to separate features of the image based on the intensity values

#a good resource to checkout: https://docs.opencv.org/3.4/db/d8e/tutorial_threshold.html

def display(img, mapp='gray'):
    plt.imshow(img, cmap=mapp, vmin=0, vmax=255)
    plt.show()

def binary_threshold(image):
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY) #threshold, #max_value
    # display(thresh1)
    cv2.imwrite('./binary_threshold.png',thresh1)


def binary_threshold_inverse(image):
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    # display(thresh2)
    cv2.imwrite('./binary_threshold_inv.png',thresh2)

def binary_threshold_truncate(image):
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC) #if gretaer than threshold: 127(threshold) else ignore max_value
    # display(thresh2)
    cv2.imwrite('./binary_threshold_trunc.png',thresh2)

def binary_threshold_to_zero(image):
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO) #if less than threshold: 0 else ignore max_value
    # display(thresh2)
    cv2.imwrite('./binary_threshold_to_zero.png',thresh2)

def binary_threshold_to_zero_inverse(image):
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
    # display(thresh2)
    cv2.imwrite('./binary_threshold_to_zero_inv.png',thresh2)
















if __name__ == '__main__':
    img_filename= r'images/rainbow.jpg'
    img = cv2.imread(img_filename,0) #read the image in B/W format
    display(img)
    while True:
        cv2.imshow('golden_retriever',img)
        #if i waited for 1 sec and i pressed q then break
        if cv2.waitKey(1) & 0xFF == ord('a'): #haxadecimal constant
            break

    cv2.destroyAllWindows()
    binary_threshold(img)

    binary_threshold_inverse(img)

    binary_threshold_truncate(img)

    binary_threshold_to_zero(img)

    binary_threshold_to_zero_inverse(img)
