##
# This file contains various functions related to working with features
##

import cv2
import numpy as np
from common import *

# Return binary Canny-edge-detected image
def getEdges(image, blur=3, threshold1=100, threshold2=200):
    if (blur > 0):
        blurImage = cv2.GaussianBlur(image, (blur,blur), 0)
    else:
        blurImage = image.copy()
    #showImage(blurImage, 'blur')
    bwImage = cv2.cvtColor(blurImage, cv2.COLOR_BGR2GRAY)
    cannyImage = cv2.Canny(bwImage,threshold1,threshold2)
    #showImage(cannyImage, 'canny')
    return cannyImage

# Return all contours that have no parents and have at least one child
def getParentContours(contours, hierarchy, childRequired=False):
    cardContours = []
    indices = []
    hierarchy = hierarchy[0]
    for i in xrange(len(contours)):
        # contour must have no parent
        if (hierarchy[i][3] == -1 and (not childRequired or hierarchy[i][2] != -1)):
            cardContours.append(contours[i])
            indices.append(i)
    return (cardContours, indices)

# Try approxPolyDP with multiple errors until we receive
# a polygon with 4 sides
# TODO: Instead of getting polygon with 4 sides, get polygon with most
# sides where 4 longest sides are significantly longer than all other
# sides. For some reason this seems to be way more accurate
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

    # didn't win, so drop to minError and get quad from intersections of 4 longest sides
    # TODO
    return []    


# Get quadrilaterals from contours by fitting polygons
def getQuadsFromContoursPoly(contours):
    quads = []
    for contour in contours:
        quad = getQuadFromContourPoly(contour)
        if len(quad) == 4:
            quads.append(quad)
    return quads

# Detect lines in contour image using hough transform
# TODO: not finished
def getQuadFromContourHough(contourImage):
    lines = cv2.HoughLinesP(contourImage, 1, np.pi/180, 70, minLineLength=30, maxLineGap=10)
    if lines != None:
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(contourImage, (x1, y1), (x2, y2), 255,5)
    showImage(contourImage)

    #lines = cv2.HoughLines(contourImage, 1, 0.5*np.pi/180, 50)
    #for line in lines[0]:
    #    rho = line[0]
    #    theta = line[1]
    #    a = math.cos(theta)
    #    b = math.sin(theta)
    #    x0 = a*rho
    #    y0 = b*rho
    #    x1 = int(x0+1000*(-b))
    #    y1 = int(y0+1000*a)
    #    x2 = int(x0-1000*(-b))
    #    y2 = int(y0-1000*a)
    #    cv2.line(contourImage, (x1, y1), (x2, y2), 255, 1)
    #showImage(contourImage)


    return [1,1,1,1]

# Detect quadrilaterals in contours using hough transforms
def getQuadsFromContoursHough(contours, image):
    quads = []
    for i in xrange(len(contours)):
        image[:,:,:] = 0
        cv2.drawContours(image, contours, i, 255)
        showImage(image)
        quad = getQuadFromContourHough(image)
        if len(quad) == 4:
            quads.append(quad)
    return quads


def getParentContoursShapeAndCount(image):
    # high blur + high detection rate for canny to get only really strong edges
    cannyEdges = getEdges(image, blur=15, threshold1=10, threshold2=30)
    # dilate so that contours are continous at sharp corners
    #showImage(cannyEdges, 'before')
    #cannyEdges  = cv2.dilate(cannyEdges, np.ones((7,7),np.uint8))
    cannyEdges = cv2.morphologyEx(cannyEdges, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))
    #showImage(cannyEdges, 'after')
    
    contours, hierarchy = cv2.findContours(cannyEdges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    (cardContours, indices) = getParentContours(contours, hierarchy, childRequired=True)


    image1 = np.zeros((image.shape[0],image.shape[1],3), np.uint8)
    cv2.drawContours(image1, contours, -1, [0,0,255])
    #showImage(image1, 'contours1')

    cv2.drawContours(image1, cardContours, -1, [255,0,0])
    #showImage(image1, 'contours2')

    cv2.drawContours(image1, [cardContours[0]], -1, [0,255,0])
    #showImage(image1, 'contours3')


    contour = cardContours[0]

    # Hu-moments are translation invariant, so this isn't necessary
    # at the moment. Nice for visualization though
    minX = min(contour[:,0,0])
    minY = min(contour[:,0,1])
    contour -= [minX, minY]

    return (contour,len(cardContours))
