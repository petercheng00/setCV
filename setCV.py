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
    return matchedSets

def matchCardCounts(cards):
    counts = []
    for card in cards:
        counts.append(card.count)

def matchCards(cards):
    return (matchCardColors(cards),
            matchCardCounts(cards),
            [],
            [])

def main():
    origImage = cv2.imread("sample.jpg")        
    showImage(origImage, 'orig')

    cannyEdges = getEdges(origImage)
    showImage(cannyEdges, 'canny')

    contours, hierarchy = cv2.findContours(cannyEdges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    (cardContours, indices) = getParentContours(contours, hierarchy, childRequired = True)

    cardQuads = getQuadsFromContoursPoly(cardContours)

    cards = createCards(cardQuads, origImage)

    (colorDict, countDict, shapeDict, fillDict) = matchCards(cards)
    print colorDict
    print countDict
            


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
