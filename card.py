import numpy as np
import cv2

class Card:
    imageSize = [440, 600]

    def __init__(self, id):
        self.id = id

        #Attributes
        self.color = None
        self.shape = None
        self.count = None
        self.fill = None

        # Image of card, without perspective
        self.image = None

        # Coordinates of card in original image, in order of 
        # UL, UR, LR, LL, if the card were in portrait mode.
        # We assume cards are 180-degree rotation invariant.
        self.origCoords = None
        
    # Set coordinates. Assume coordinates are passed in
    # adjacent order, and shift by 1 if necessary
    # NOTE: this assumes all long edges in perspective
    # view correspond to all long edges in not-perspective
    # view which is very not guaranteed
    def setCoords(self, coords):
        if np.linalg.norm(coords[1]-coords[0]) > np.linalg.norm(coords[2]-coords[1]):
            # 2 because array is flattened
            coords = np.roll(coords, 2)
        self.origCoords = coords
        

    def setImage(self, origImage, coords):
        self.setCoords(coords)
        w = self.imageSize[0]
        h = self.imageSize[1]
        dstPoints = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        transform = cv2.getPerspectiveTransform(self.origCoords.astype(np.float32), dstPoints)
        self.image = cv2.warpPerspective(origImage, transform, (w, h))
