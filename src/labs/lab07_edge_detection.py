# import cv2 as cv
# import numpy as np
#
# img = cv.imread('lab007task1 (1).png', cv.IMREAD_GRAYSCALE)
#
# sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
# sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
#
# mag = cv.magnitude(sobel_x, sobel_y)
#
# threshold = np.percentile(mag, 70)
# _, mag_thresh = cv.threshold(mag, threshold, 255, cv.THRESH_BINARY)
#
# phase = np.arctan2(sobel_y, sobel_x)
#
# mask_45 = np.logical_and(phase > np.radians(45 - 5), phase < np.radians(45 + 5))
# mask_90 = np.logical_and(phase > np.radians(90 - 5), phase < np.radians(90 + 5))
#
# mask_45_mag = np.logical_and(mask_45, mag_thresh > 0)
# mask_90_mag = np.logical_and(mask_90, mag_thresh > 0)
#
# mask_45_mag = (mask_45_mag * 255).astype(np.uint8)
# mask_90_mag = (mask_90_mag * 255).astype(np.uint8)
#
# # Save the images
# cv.imshow('magnitude_threshold.png', mag_thresh)
# cv.imshow('lines_45_degrees.png', mask_45_mag)
# cv.imshow('lines_90_degrees.png', mask_90_mag)
#
# cv.waitKey()

# Task 02

# import cv2 as cv
# import numpy as np
#
# img = cv.imread("lab007task 2,3 (1).png")
#
# mean = np.mean(img)
# median = np.median(img)
#
# _, thresh_mean = cv.threshold(img, mean, 255, cv.THRESH_BINARY)
# _, thresh_median = cv.threshold(img, median, 255, cv.THRESH_BINARY)
#
# cv.imshow('Mean Threshold Image', thresh_mean)
# cv.imshow('Median Threshold Image', thresh_median)
# cv.waitKey(0)


# Task 03
#
# import cv2 as cv
# import numpy as np
#
# img = cv.imread("lab007task 2,3 (1).png", cv.IMREAD_GRAYSCALE)
# np.pad(img,(1,1),mode='mean')
# new_img = np.zeros_like(img)
#
# for i in range(new_img.shape[0]):
#     for j in range(new_img.shape[1]):
#         neighborhood = img[i:i + 3, j:j + 3]
#
#         # Calculate the mean of the neighborhood
#         mean = np.mean(neighborhood)
#
#         # Apply the threshold
#         if img[i, j] < mean - 2:
#             new_img[i, j] = 0
#         else:
#             new_img[i, j] = 255
#
# cv.imshow("Local mean Threshold image", new_img)
#
# cv.waitKey(0)

# Task 04

# import cv2 as cv
# import numpy as np
#
# img = cv.imread("IMD002.bmp", cv.IMREAD_GRAYSCALE)
#
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#
# flags = cv.KMEANS_RANDOM_CENTERS
#
# compactness, labels, centers = cv.kmeans(img.reshape(-1,1).astype(np.float32), 2, None, criteria, 10, flags)
#
# segmented_image = centers[labels.flatten()].reshape(img.shape).astype(np.uint8)
#
# cv.imshow("Clustered image", segmented_image)
# cv.waitKey(0)