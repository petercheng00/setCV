import cv2
import numpy as np
import math
import constants

from card import Card
from features import *
from sets import Set

# webcam index
# -1 means auto select
camIndex = -1

def createCards(cardQuads, origImage):
    cards = []
    id = 0
    for quad in cardQuads:
        card = Card(id, origImage, quad)
        cards.append(card)
        id += 1
    return cards

def sameCardColor(hist1, hist2):
    histDiff = cv2.compareHist(hist1, hist2, 0)
    return histDiff > constants.color_similarity_threshold

def sameCardCount(count1, count2):
    return count1 == count2

def sameCardShape(shape1, shape2):
    val = cv2.matchShapes(shape1, shape2, 1, 0.0)
    print val

    image1 = np.zeros((200,400,3), np.uint8)
    cv2.drawContours(image1, [shape1], 0, 255)
    showImage(image1, 'contour1', wait=False)
    image2 = np.zeros((200,400,3), np.uint8)
    cv2.drawContours(image2, [shape2], 0, 255)
    showImage(image2, 'contour2')

    return val < constants.shape_similarity_threshold


def matchCardAttributes(cards, attribute, equalityTest):
    unmatched = list(xrange(len(cards)))
    matched = []

    while len(unmatched) > 0:
        i = unmatched.pop(0)
        newSet = [i]
        value1 = getattr(cards[i], attribute)
        for j in reversed(unmatched):
            value2 = getattr(cards[j], attribute)
            if equalityTest(value1, value2):
                newSet.append(j)
                unmatched.remove(j)
        matched.append(newSet)

    matchedSets = []
    for match in matched:
        matchedSets.append(set(match))
    return matchedSets

def matchCards(cards):
    return (matchCardAttributes(cards, 'color', sameCardColor),
            matchCardAttributes(cards, 'count', sameCardCount),
            matchCardAttributes(cards, 'shape', sameCardShape),
            [])

def main():
    origImage = cv2.imread("sample.jpg")        
    #showImage(origImage, 'orig')

    cannyEdges = getEdges(origImage)
    #showImage(cannyEdges, 'canny')

    contours, hierarchy = cv2.findContours(cannyEdges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    (cardContours, indices) = getParentContours(contours, hierarchy, childRequired = True)

    cardQuads = getQuadsFromContoursPoly(cardContours)

    cards = createCards(cardQuads, origImage)

    (colorDict, countDict, shapeDict, fillDict) = matchCards(cards)
    print 'colors'
    print colorDict
    print 'counts'
    print countDict
    print 'shapes'
    print shapeDict
    
    #for card in cards:
    #    showImage(card.image, 'asdf')


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
