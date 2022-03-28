# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 20:12:33 2022

@author: hp
"""

# implementation
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
#from Computer_Vision.Sobel_Edge_Detection.convolution 
import convolution
#from Computer_Vision.Sobel_Edge_Detection.gaussian_smoothing 
import gaussian_blur
 
filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "-window.jpg", required=True, help='\Desktop\Image_ASS2')
args = vars(ap.parse_args())
 
image = cv2.imread(args["image"])
image = gaussian_blur(image, 9, verbose=True)
def sobel_edge_detection(image, filter, verbose=False):
    new_image_x = convolution(image, filter, verbose)
 
    if verbose:
        plt.imshow(new_image_x, cmap='gray')
        plt.title("Horizontal Edge")
        plt.show()
 
    new_image_y = convolution(image, np.flip(filter.T, axis=0), verbose)
 
    if verbose:
        plt.imshow(new_image_y, cmap='gray')
        plt.title("Vertical Edge")
        plt.show()
 
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
 
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
 
    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()
 
    return gradient_magnitude
 
sobel_edge_detection(image, filter, verbose=True)


   