import cv2

# Show image in the window with the specified name
# If window does not exist, will be created
def showImage(im, name):
    cv2.imshow(name, im)
    cv2.waitKey()
