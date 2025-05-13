# lab 04: Image Enhancement
import cv2 as cv
import numpy as np

# 1. Image Transforms

#
# # 1.1. Negative Transformation
#
# def negative_transform(colored_image):
#     return 255 - colored_image
#
#
# # 1.2. Logarithm Transformation
#
#
# filename = 'Fig0241(a)(einstein low contrast).tif'
# einstein = cv.imread(filename, cv.IMREAD_GRAYSCALE)
#
# c = 255 / np.log(1 + np.max(einstein))
# einstein_log = np.uint8(c * np.log(1 + einstein))
# cv.imshow('Einstein Logarithm', einstein_log)
# cv.waitKey()


# Task 2:
# Less than mean        0
# Greater than mean     255

# filename = 'Fig0241(a)(einstein low contrast).tif'
# einstein = cv.imread(filename, cv.IMREAD_GRAYSCALE)
# mean = np.mean(einstein)
# new_image = np.zeros_like(einstein)

# for i in range(einstein.shape[0]):
#     for j in range(einstein.shape[1]):
#         if einstein[i, j] - mean > 20 or einstein[i, j] - mean < -20:
#             new_image[i, j] = 255
#         else:
#             new_image[i, j] = 0

# cv.imshow('Einstein', new_image)
# cv.waitKey()


# # Task 03: Power Law
# filename = 'Fig0241(a)(einstein low contrast).tif'
# colored_image = cv.imread(filename,cv.IMREAD_GRAYSCALE)

# # s=255*[(r/255)^ γ]
# # 3.1. gamma = 0.2
# gamma1 = 0.2

# image1 = np.uint8(255 * ((colored_image/255) ** gamma1))
# cv.imshow('Einstein when gamma = 0.2', image1)

# # 3.2. gamma = 0.5
# gamma1 = 0.5
# image1 = np.uint8(255 * ((colored_image/255) ** gamma1))
# cv.imshow('Einstein when gamma = 0.5', image1)

# # 3.3. gamma = 1.2
# gamma1 = 1.2
# image1 = np.uint8(255 * ((colored_image/255) ** gamma1))
# cv.imshow('Einstein when gamma = 1.2', image1)

# # 3.4. gamma = 1.8
# gamma1 = 1.8
# image1 = np.uint8(255 * ((colored_image/255) ** gamma1))
# cv.imshow('Einstein when gamma = 1.8', image1)
# cv.waitKey()




# # Task 04: Gray Level Slicing
# filename = 'Fig0241(a)(einstein low contrast).tif'
# colored_image = cv.imread(filename,cv.IMREAD_GRAYSCALE)

# image2 = np.zeros_like(colored_image)
# for i in range(colored_image.shape[0]):
#     for j in range(colored_image.shape[1]):
#        if(100 < colored_image[i,j] and colored_image[i,j] < 210):
#            image2[i,j] = 210
#        else:
#             image2[i,j] = colored_image[i,j]


# cv.imshow('Einstein 210 between 100 and 200', image2) 
# cv.waitKey()


# Task 05: Histogram
import matplotlib.pyplot as plt
filename = 'Fig0241(a)(einstein low contrast).tif'
image =cv.imread(filename, 0)
histogram = np.zeros(256)
r = image.shape[0]
c= image.shape[1]
for i in range(r):
    for j in range(c):
        value = int((image[i,j]))
        histogram[value]= 1 + histogram[value]
plt.bar(range(len(histogram)), histogram)
plt.show()
