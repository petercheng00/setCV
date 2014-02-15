import numpy as np
import cv2
import random
import time
from matplotlib import pyplot as plt

class Card:
    # resolution to use for card size. could be tweaked
    imageSize = [440, 600]

    def __init__(self, id, origImage, coords):
        self.id = id

        # Coordinates of card in original image, in order of 
        # UL, UR, LR, LL, if the card were in portrait mode.
        # We assume cards are 180-degree rotation invariant.
        self.origCoords = self.getCoords(coords)

        # Image of card, without perspective
        self.image = self.getImage(origImage)

        # Attributes
        # self.hueHistogram = self.getHueHistogram()
        self.hueSatHistogram = self.getHueSatHistogram()
        self.count = None
        self.shape = None
        self.fill = None

        
    # Set coordinates. Assume coordinates are passed in
    # adjacent order, and shift by 1 if necessary
    # NOTE: this assumes all long edges in perspective
    # view correspond to all long edges in not-perspective
    # view which is very not guaranteed
    def getCoords(self, coords):
        if np.linalg.norm(coords[1]-coords[0]) > np.linalg.norm(coords[2]-coords[1]):
            # 2 because array is flattened
            coords = np.roll(coords, 2)
        return coords

    def getImage(self, origImage):
        w = self.imageSize[0]
        h = self.imageSize[1]
        dstPoints = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        transform = cv2.getPerspectiveTransform(self.origCoords.astype(np.float32), dstPoints)
        return cv2.warpPerspective(origImage, transform, (w, h))

    # convert to HSV get histogram on hue
    def getHueHistogram(self):
        hsvImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hueHist = cv2.calcHist([hsvImage], [0], None, [180], [0,180]) 
        return hueHist

    # convert to HSV get 2d histogram on hue and saturation
    def getHueSatHistogram(self):
        hsvImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hueSatHist = cv2.calcHist([hsvImage], [0,1], None, [180,256], [0,180, 0,256])

        # remove low saturation which corresponds to white
        hueSatHist[:,0:50] = 0
        return hueSatHist
