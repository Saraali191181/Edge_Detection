# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 21:38:16 2022

@author: hp
"""

import cv2

import numpy as np

 
image = cv2.imread('Desktop\Image_ASS2\window.pjp')

 
# Print error message if image is null

if image is None:

    print('Could not read image')

 

# Apply identity kernel

kernel1 = np.array([[0, 0, 0],

                    [0, 1, 0],

                    [0, 0, 0]])

 

identity = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)


cv2.imshow('Desktop\Image_ASS2\window.pjp', image)

cv2.imshow('Desktop\Image_ASS2\window.pjp', identity)
     

cv2.waitKey()

cv2.imwrite('Desktop\Image_ASS2\window.pjp', identity)

cv2.destroyAllWindows()
 

# Apply blurring kernel

kernel2 = np.ones((5, 5), np.float32) / 25

img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)

 
cv2.imshow('Desktop\Image_ASS2\window.pjp', image)

cv2.imshow('Desktop\Image_ASS2\window.pjp', img)

     
cv2.waitKey()

cv2.imwrite('Desktop\Image_ASS2\window.pjp', img)

cv2.destroyAllWindows()
