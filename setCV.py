import cv2
import numpy as np

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

def getRectsFromContours(contours):
    rects = []
    for contour in contours:
        rects.append(cv2.approxPolyDP(contour, 0.1, True))
    return rects

def main():
    origImage = cv2.imread("sampleSetImage.jpg")
    showImage(origImage)

    cannyEdges = getEdges(origImage)
    showImage(cannyEdges)

    cardContours = getParentContours(cannyEdges)
    contourImage = np.zeros(cannyEdges.shape)
    cv2.drawContours(contourImage, cardContours, -1, 1)
    showImage(contourImage)

    cardRects = getRectsFromContours(cardContours)

    cv2.fillPoly(origImage, cardRects, (255,0,0))
    showImage(origImage, "bla")


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
