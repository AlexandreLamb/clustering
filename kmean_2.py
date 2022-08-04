import cv2
import numpy as np

# read input and convert to range 0-1
image = cv2.imread('1.jpg')
h, w, c = image.shape

# reshape to 1D array
image_2d = image.reshape(h*w, c).astype(np.float32)

# set number of colors
numcolors = 5
numiters = 10
epsilon = 1
attempts = 10

# do kmeans processing
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, numiters, epsilon)
ret, labels, centers = cv2.kmeans(image_2d, numcolors, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

# reconstitute 2D image of results
centers = np.uint8(centers)
newimage = centers[labels.flatten()]
newimage = newimage.reshape(image.shape)
cv2.imwrite("eye_kmeans.png", newimage)
cv2.imshow('new image', newimage)
cv2.waitKey(0)

k = 0
for center in centers:
    # select color and create mask
    #print(center)
    layer = newimage.copy()
    mask = cv2.inRange(layer, center, center)

    # apply mask to layer 
    layer[mask == 0] = [0,0,0]
    cv2.imshow('layer', layer)
    cv2.waitKey(0)


    # save kmeans clustered image and layer 
    cv2.imwrite("eye{0}.png".format(k), layer)
    k = k + 1
