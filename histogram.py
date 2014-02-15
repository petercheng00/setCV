##
# This file contains functions related to working with histograms
##

import cv2

# convert to HSV get histogram on hue
def getHueHistogram(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hueHist = cv2.calcHist([hsvImage], [0], None, [180], [0,180]) 
    return hueHist

# convert to HSV get 2d histogram on hue and saturation
def getHueSatHistogram(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hueSatHist = cv2.calcHist([hsvImage], [0,1], None, [180,256], [0,180, 0,256])

    # remove low saturation which corresponds to white
    hueSatHist[:,0:50] = 0
    return hueSatHist
