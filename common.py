import cv2
import sys

# Show image in the window with the specified name
# If window does not exist, will be created
def showImage(im, name, wait=True):
    sys.stdout.flush()
    cv2.imshow(name, im)
    if wait:
        key = cv2.waitKey()
        if key == 27: # exit on ESC
            sys.exit()
