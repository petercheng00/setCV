import cv2
import numpy as np
import math
import itertools

from settings import *
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
        if card.valid:
            cards.append(card)
            id += 1
    return cards

def filterQuadSizes(quads):
    quadPerimeters = [np.linalg.norm(quad[1]-quad[0]) + \
                      np.linalg.norm(quad[2]-quad[1]) + \
                      np.linalg.norm(quad[3]-quad[2]) + \
                      np.linalg.norm(quad[0]-quad[3]) for quad in quads]
    perimeterMedian = np.median(quadPerimeters)
    perimeterStd = np.std(quadPerimeters)
    if perimeterStd > perimeterMedian * 0.25:
        valid = np.array([abs(perimeter - perimeterMedian) < 2*perimeterStd for perimeter in quadPerimeters])
        return quads[valid]
    else:
        return quads

def sameCardColor(hist1, hist2):
    histIntersect = cv2.compareHist(hist1, hist2, 2)
    sumHist1 = sum(sum(hist1))
    sumHist2 = sum(sum(hist2))
    if DEBUGCOLORS:
        print str(histIntersect / sumHist1 > color_similarity_threshold or \
        histIntersect / sumHist2 > color_similarity_threshold) + \
        ' hist ratio = ' + str(histIntersect/sumHist1) + ', ' + str(histIntersect/sumHist2)
        showImage(hist1, 'hist1', wait=False)
        showImage(hist2, 'hist2')

    return histIntersect / min(sumHist1, sumHist2) > color_similarity_threshold

def sameCardCount(count1, count2):
    return count1 == count2

def sameCardShape(shape1, shape2):
    #val = cv2.matchShapes(shape1, shape2, 1, 0.0)
    #print val
    (maxX1, maxY1) = (max(shape1[:,0,0]), max(shape1[:,0,1]))
    (minX1, minY1) = (min(shape1[:,0,0]), min(shape1[:,0,1]))
    (maxX2, maxY2) = (max(shape2[:,0,0]), max(shape2[:,0,1]))
    (minX2, minY2) = (min(shape2[:,0,0]), min(shape2[:,0,1]))

    maxXDiff = max(maxX1 - minX1, maxX2 - minX2)
    maxYDiff = max(maxY1 - minY1, maxY2 - minY2)

    image1 = np.zeros((maxYDiff,maxXDiff), np.uint8)
    cv2.drawContours(image1, [shape1], 0, 255, offset=(-minX1, -minY1))
    image1 = cv2.dilate(image1, np.ones((17,17), np.uint8))
    if DEBUGSHAPES:
        showImage(image1, 'contour1', wait=False)

    image2 = np.zeros((maxYDiff,maxXDiff), np.uint8)
    cv2.drawContours(image2, [shape2], 0, 255, offset=(-minX2, -minY2))
    image2 = cv2.dilate(image2, np.ones((17,17), np.uint8))
    if DEBUGSHAPES:
        showImage(image2, 'contour2', wait=False)


    intersectImage = cv2.bitwise_and(image1, image2)
    intersectCount = float(cv2.countNonZero(intersectImage))
    count1 = cv2.countNonZero(image1)
    count2 = cv2.countNonZero(image2)
    if DEBUGSHAPES:
        print str(intersectCount / count1 > shape_similarity_threshold or \
                  intersectCount / count2 > shape_similarity_threshold) + \
            ' intersect ratio = ' + str(intersectCount/count1) + ', ' + str(intersectCount/count2)
        showImage(intersectImage, 'and')
    
    return intersectCount / min(count1, count2) > shape_similarity_threshold

def sameCardFill(fillPct1, fillPct2):
    return getFillAmount(fillPct1) == getFillAmount(fillPct2)

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
        matchedSets.append(frozenset(match))
    return matchedSets

def matchCards(cards):
    return (matchCardAttributes(cards, 'color', sameCardColor),
            matchCardAttributes(cards, 'count', sameCardCount),
            matchCardAttributes(cards, 'shape', sameCardShape),
            matchCardAttributes(cards, 'fillPct', sameCardFill))

def isSet(cardIds, matchSets):
    for matchSet in matchSets:

        def findSet(item):
            for mSet in matchSet:
                if item in mSet:
                    return mSet
            return None

        cardSets = map(findSet, cardIds)
        if all(cs == cardSets[0] for cs in cardSets):
            continue
        if len(cardSets) == len(set(cardSets)):
            continue
        return False
    return True
            

def getSets(cards, matchSets):
    sets = []
    for combination in itertools.combinations(cards,3):
        cardIds = map(lambda x:x.id, combination)
        if isSet(cardIds, matchSets):
            sets.append(combination)
    return sets

def detectSet(origImage):
    cannyEdges = getEdges(origImage)

    contours, hierarchy = cv2.findContours(cannyEdges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    (cardContours, indices) = getParentContours(contours, hierarchy, childRequired = True)

    cardQuads = getQuadsFromContoursPoly(cardContours)

    cardQuads = filterQuadSizes(cardQuads)

    cards = createCards(cardQuads, origImage)

    (colorDict, countDict, shapeDict, fillDict) = matchCards(cards)
    
    cardSets = getSets(cards, (colorDict, countDict, shapeDict, fillDict))
    
    #contourImage = origImage.copy()
    #cv2.drawContours(contourImage, contours, -1, 255, thickness=5)
    #cv2.imshow('canny', cannyEdges)
    #cv2.imshow('contours', contourImage)
    cardsImage = origImage.copy()
    for card in cards:
        cv2.polylines(cardsImage, [card.origCoords], True, (255,0,0), thickness=10)
    cv2.imshow('cards', cardsImage)


def detectSetDebug(origImage):
    cannyEdges = getEdges(origImage)

    contours, hierarchy = cv2.findContours(cannyEdges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    (cardContours, indices) = getParentContours(contours, hierarchy, childRequired = True)

    cardQuads = getQuadsFromContoursPoly(cardContours)

    cardQuads = filterQuadSizes(cardQuads)

    cards = createCards(cardQuads, origImage)

    (colorDict, countDict, shapeDict, fillDict) = matchCards(cards)
    
    cardSets = getSets(cards, (colorDict, countDict, shapeDict, fillDict))

    for cardSet in cardSets:
        copyImage = origImage.copy()
        for i in xrange(3):
            cv2.polylines(copyImage, [cardSet[i].origCoords], True, (255,0,0), thickness=10)
        showImage(copyImage)

    if DEBUG:
        showImage(origImage, 'orig')
        showImage(cannyEdges, 'canny')
        contourImage = origImage.copy()
        cv2.drawContours(contourImage, cardContours, -1, 255, thickness=5)
        showImage(contourImage, 'contours')
        print 'num cards: ' + str(len(cards))
        cardsImage = origImage.copy()
        for card in cards:
            cv2.polylines(cardsImage, [card.origCoords], True, (255,0,0), thickness=10)
        showImage(cardsImage, 'all cards')
        #for card in cards:
        #    showImage(card.image, str(card.id), wait=True, replaceAll=True)
        print 'colors'
        print colorDict
        print 'counts'
        print countDict
        print 'shapes'
        print shapeDict
        print 'fills'
        print fillDict
        for cardSet in cardSets:
            print map(lambda x:x.id, cardSet)
            showImage(cardSet[0].image, 'card1', wait=False)
            showImage(cardSet[1].image, 'card2', wait=False)
            showImage(cardSet[2].image, 'card3')

def webcam():
    cv2.namedWindow('webcam')
    vc = cv2.VideoCapture(camIndex)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow('webcam', frame)
        detectSet(frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow("preview")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        detectSet(cv2.imread(sys.argv[1]))
    else:
        webcam()
