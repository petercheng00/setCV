import cv2
import sys

# Show image in the window with the specified name
# If window does not exist, will be created
def showImage(im, name='image', wait=True, replaceAll=False):
    sys.stdout.flush()
    if replaceAll:
        cv2.destroyAllWindows()
    cv2.imshow(name, im)
    if wait:
        key = cv2.waitKey()
        if key == 27: # exit on ESC
            sys.exit()
