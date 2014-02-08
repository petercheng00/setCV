import cv2
import numpy as np
import math

# webcam index
# -1 means auto select
camIndex = -1

# window names
SOURCE = "source"

def showImage(im, name=SOURCE):
    cv2.imshow(name, im)
    cv2.waitKey()

def getEdges(image):
    blurImage = cv2.GaussianBlur(image, (3,3), 0)
    bwImage = cv2.cvtColor(blurImage, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(bwImage,100,200)
    
def getParentContours(bwImage):
    contours, hierarchy = cv2.findContours(bwImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cardContours = []
    for i in xrange(len(contours)):
        # contour must have no parent, and must have at least 1 child
        if (hierarchy[0][i][3] == -1 and hierarchy[0][i][2] != -1):
            cardContours.append(contours[i])
    return cardContours

# Try approxPolyDP with multiple errors until we receive
# a polygon with 4 sides
def getQuadFromContourPoly(contour):
    iters = 0
    maxIters = 100
    minError = 0.1
    maxError = 10
    currError = 2.5
    while iters < maxIters:
        polygon = cv2.approxPolyDP(contour, currError, True)

        if (len(polygon) == 4):
            return polygon
        elif (len(polygon) > 4):
            minError = currError
            currError = (currError + maxError) / 2
        else:
            maxError = currError
            currError = (minError + currError) / 2
        iters += 1

    #didn't win, so drop to minError and get quad from intersections of 4 longest sides
    #todo

    return []    

def getQuadsFromContoursPoly(contours):
    quads = []
    for contour in contours:
        quad = getQuadFromContourPoly(contour)
        if len(quad) == 4:
            quads.append(quad)
    return quads

def getQuadFromContourHough(contour):
    contourImage = np.zeros(imageSize)
    cv2.drawContours(contourImage, contour, -1, 1)
    lines = cv2.HoughLinesP(contourImage, 1, math.pi/180, 80)
    print lines


def getRectsFromQuads(quads):
    return quads

def main():
    origImage = cv2.imread("sampleSetImage.jpg")
    showImage(origImage)

    cannyEdges = getEdges(origImage)
    showImage(cannyEdges)

    cardContours = getParentContours(cannyEdges)
    #contourImage = np.zeros(cannyEdges.shape)
    #cv2.drawContours(contourImage, cardContours, -1, 1)
    #showImage(contourImage)

    cardQuads = getQuadsFromContoursPoly(cardContours)
    cardRects = getRectsFromQuads(cardQuads)

    
    cv2.polylines(origImage, cardRects, True, (255,0,0),3)
    showImage(origImage, "BLA")



if __name__ == "__main__":
    main()


def webcam():
    cv2.namedWindow(SOURCE)
    vc = cv2.VideoCapture(camIndex)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow(SOURCE, frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow("preview")