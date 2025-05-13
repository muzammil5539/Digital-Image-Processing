# Figure shows an image of the rear of a vehicle. Your  task is to the
# use image processing  algorithm for finding rectangles whose sizes
# makes them suitable candidates for license plates.

import cv2 as cv

image1 = cv.imread('lab final.tif', cv.IMREAD_GRAYSCALE)
image = cv.threshold(image1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
# connected components with stats to find the bounding box of the objects
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
# iterate through the bounding boxes
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    # filter the bounding boxes based on the area
    print(area)
    if 500 <= area <= 700:
        cv.rectangle(image1, (x, y), (x + w, y + h), (255, 255, 255), 2)
cv.imshow('image2', image1)
cv.waitKey()
