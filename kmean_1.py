import cv2
import numpy as np
import matplotlib.pyplot as plt
for i in range(1,6):
    img = cv2.imread(str(i)+'.jpg') 

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    pixel_vals =img.reshape((-1,3))

    pixel_vals = np.float32(pixel_vals)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.85)

    k= 5

    retval, labels, centers = cv2.kmeans(pixel_vals,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)


    clustered_img = labels.reshape((img.shape[0], img.shape[1]))

    clusters_to_0 = [1,2,4]

    for c in clusters_to_0:
        clustered_img[clustered_img == c] = -1

    clustered_img[clustered_img!=-1] = 1
    clustered_img[clustered_img==-1] = 0

    clustered_img
    plt.imshow(clustered_img)
    plt.show()