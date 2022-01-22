import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(img, mapp='gray'):
    plt.imshow(img, cmap=mapp)
    plt.show()


def overlay(imgp1,imgp2):
    img1 = cv2.imread(imgp1)
    img2 = cv2.imread(imgp2)
    img2 =cv2.resize(img2,(300,300))

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    large_img = img1
    small_img = img2

    # x_offset=0
    # y_offset=0

    large_img[0:0+small_img.shape[0], 0:0+small_img.shape[1]] = small_img
    display(large_img)

def vanilla_blending(imgp1,imgp2):
    img1= cv2.imread(imgp1)
    img2= cv2.imread(imgp2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1 =cv2.resize(img1,(1200,1200))
    img2 =cv2.resize(img2,(1200,1200))
    blended = cv2.addWeighted(src1=img1,alpha=0.7,src2=img2,beta=0.3,gamma=0)
    display(blended)

def blend_image(imgp1,imgp2):
    img1= cv2.imread(imgp1)
    img2= cv2.imread(imgp2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2= cv2.resize(img2,(0,0), img2, 0.3, 0.3)

    # display(img1)
    # display(img2)
    print("the shape of img1:> ",img1.shape) #hwd
    print("the shape of img2:> ",img2.shape) #hwd


    #create a ROI of the img1
    y_offset= img1.shape[0] - img2.shape[0] #height - height
    x_offset= img1.shape[1] - img2.shape[1] #width - width

    roi= img1[y_offset: , x_offset:,:]
    print(f"shape of roi:> {roi.shape}")
    # display(roi)

    #########################################################
    # creating a mask
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    print(f"shape of img2gray:> {img2gray.shape}")
    display(img2gray)

    #inverse the mask---> doubt
    mask_inv = cv2.bitwise_not(img2gray) #only 2 channels
    display(mask_inv)

    #place the mask on top of img2-- so as to extract the logo part
    fg = cv2.bitwise_or(img2, img2, mask=mask_inv)
    display(fg)


    #final roi image
    final_roi= cv2.bitwise_or(roi,fg)
    display(final_roi)

    img1[y_offset: , x_offset:,:]= final_roi

    display(img1)



if __name__ == '__main__':
    img1= r'D:\Exploring-Tensorflow\classical_cv\images\golden_retriever.jpg'
    img2= r'D:\Exploring-Tensorflow\classical_cv\images\watermark_no_copy.png'
    # vanilla_blending(img1,img2)

    # overlay(img1,img2)

    blend_image(img1, img2)