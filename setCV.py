import cv2
import numpy as np
import math
from card import Card
import constants
from sets import Set

# webcam index
# -1 means auto select
camIndex = -1


def showImage(im, name):
    cv2.imshow(name, im)
    cv2.waitKey()

def getEdges(image):
    blurImage = cv2.GaussianBlur(image, (3,3), 0)
    bwImage = cv2.cvtColor(blurImage, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(bwImage,100,200)
    
def getParentContours(contours, hierarchy):
    cardContours = []
    indices = []
    hierarchy = hierarchy[0]
    for i in xrange(len(contours)):
        # contour must have no parent, and must have at least 1 child
        if (hierarchy[i][3] == -1 and hierarchy[i][2] != -1):
            cardContours.append(contours[i])
            indices.append(i)
    return (cardContours, indices)

# Try approxPolyDP with multiple errors until we receive
# a polygon with 4 sides
# TODO: Instead of getting polygon with 4 sides, get polygon with most sides where 4 longest sides are significantly longer than all other sides. For some reason this seems to be way more accurate
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

def createCards(cardQuads, origImage):
    cards = []
    id = 0
    for quad in cardQuads:
        card = Card(id, origImage, quad)
        cards.append(card)
        id += 1
    return cards

def matchCardColors(cards):
    unmatched = list(xrange(len(cards)))
    matched = []

    while len(unmatched) > 0:
        i = unmatched.pop(0)
        newColor = [i]
        
        hist1 = cards[i].hueSatHistogram
        for j in reversed(unmatched):
            hist2 = cards[j].hueSatHistogram
            # compute histogram correlation
            histDiff = cv2.compareHist(hist1, hist2, 0)
            if histDiff > constants.color_similarity_threshold:
                newColor.append(j)
                unmatched.remove(j)
        matched.append(newColor)

    matchedSets = []
    for match in matched:
        matchedSets.append(set(match))

def matchCards(cards):
    return (matchCardColors(cards),
            [],
            [],
            [])

def main():
    origImage = cv2.imread("sample.jpg")        
    hsvImage = cv2.cvtColor(origImage, cv2.COLOR_BGR2HSV)
    showImage(origImage, 'orig')

    cannyEdges = getEdges(origImage)
    #showImage(cannyEdges)

    contours, hierarchy = cv2.findContours(cannyEdges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    (cardContours, indices) = getParentContours(contours, hierarchy)

    cardQuads = getQuadsFromContoursPoly(cardContours)

    cards = createCards(cardQuads, origImage)

    (colorDict, shapeDict, countDict, fillDict) = matchCards(cards)
            


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
