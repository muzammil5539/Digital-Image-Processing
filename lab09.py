import numpy as np

def erosion(image, kernel):
    m, n = image.shape
    k, _ = kernel.shape
    pad = k // 2
    output = np.zeros((m, n))
    padded_image = np.pad(image, pad)
    for i in range(m):
        for j in range(n):
            section = padded_image[i:i+k, j:j+k]
            output[i, j] = np.min(section * kernel)
    return output

def dilation(image, kernel):
    m, n = image.shape
    k, _ = kernel.shape
    pad = k // 2
    output = np.zeros((m, n))
    padded_image = np.pad(image, pad)
    for i in range(m):
        for j in range(n):
            section = padded_image[i:i+k, j:j+k]
            output[i, j] = np.max(section * kernel)
    return output


# Task 01 Erosion and dilation of image
#
# import cv2 as cv
#
# img = cv.imread("lab009coin.png", cv.IMREAD_GRAYSCALE)
#
# # Convert the image to binary
# _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
#
# # Define the kernel
# size = int(input('Size of matrix'))
# kernel = np.ones((size,size))
#
# # Apply erosion
#
# erosion_img = erosion(img, kernel)
#
# # Apply dilation
#
# dilation_img = dilation(img, kernel)
#
# cv.imshow("Original Image", img)
# cv.imshow("Erosion Image {}x{}".format(size, size), erosion_img)
# cv.imshow("Dilation Image {}x{}".format(size, size), dilation_img)
# cv.waitKey(0)





# Lab Task 2:
# Remove the noise from Fig 2 and then fill the holes or gap between thumb impressions. You can
# apply morphological closing and opening.

#
# import cv2 as cv
# img = cv.imread("Fig0911(a)(noisy_fingerprint).tif", cv.IMREAD_GRAYSCALE)
# # Convert the image to binary
# _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# # Define the kernel
# size = int(input('Size of matrix'))
# kernel = np.ones((size,size))
#
# # Apply closing
# opening_img = dilation(erosion(img, kernel), kernel)
#
# # Apply opening
# closing_img = erosion(dilation(img, kernel), kernel)
#
# cv.imshow("Original Image", img)
# cv.imshow("Closing Image {}x{}".format(size, size), closing_img)
# cv.imshow("Opening Image {}x{}".format(size, size), opening_img)
# cv.waitKey(0)

# Lab Task 3: We have 512 *512 image of a head CT scan. Perform Gray scale 3x3 dilation and erosion on Fig 3. Also
# find Morphological gradient. Use following expression to compute gradient:
# Gradient = Dilation - Erosion
#
# import cv2 as cv
# img = cv.imread("Fig0939(a)(headCT-Vandy).tif", cv.IMREAD_GRAYSCALE)
#
# # Convert the image to binary
# _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
#
#
# # Define the kernel
# size = int(input('Size of matrix'))
# kernel = np.ones((size,size))
#
# # Apply dilation
# dilation_img = dilation(img, kernel)
#
# # Apply erosion
# erosion_img = erosion(img, kernel)
#
# # Apply gradient
# gradient = dilation_img - erosion_img
#
# cv.imshow("Original Image", img)
# cv.imshow("Dilation Image {}x{}".format(size, size), dilation_img)
# cv.imshow("Erosion Image {}x{}".format(size, size), erosion_img)
# cv.imshow("Gradient Image {}x{}".format(size, size), gradient)
# cv.waitKey(0)

# Lab Task 4: Apply top-hat transformation on the following image using given expression:
# Top-hat = Original Image - Opening

# import cv2 as cv
# img = cv.imread("Fig0940(a)(rice_image_with_intensity_gradient).tif", cv.IMREAD_GRAYSCALE)
#
# # Define the kernel
# size = int(input('Size of matrix'))
# kernel = np.ones((size,size))
#
# # Apply opening
#
# opening_img = dilation(erosion(img, kernel), kernel)
#
# # Apply top-hat transformation
#
# top_hat = img - opening_img
#
# cv.imshow("Original Image", img)
# cv.imshow("Top-hat Image {}x{}".format(size, size), top_hat)
# cv.waitKey(0)






