# '''     
#     Task 01
#         Load these RGB images 

  
# Convert it to HSV color space using above-mentioned equations. Display each channel (Hue, saturation and intensity/value) separately

# '''
# import cv2 as cv
# import numpy as np

# filename = 'lab10task1.tif'

# img = cv.imread(filename, cv.IMREAD_COLOR)

# b, g, r = cv.split(img)

# b = b.astype('float64')
# g = g.astype('float64')
# r = r.astype('float64')

# hue = np.arccos(0.5 * ((r - g) + (r - b)) / np.sqrt((r - g) ** 2 + (r - b) * (g - b)))

# sat = 1 - (3 / (r + g + b)) * np.minimum(r, np.minimum(g, b))

# value = (r + g + b) / 3.0

# hue = np.uint8((hue / np.pi) * 180)  
# sat = np.uint8(sat * 255)
# value = np.uint8(value)

# cv.imshow("Hue", hue)
# cv.imshow("Saturation", sat)
# cv.imshow("Value", value)
# cv.waitKey(0)
# cv.destroyAllWindows()



# '''
#     Task 02
#     Read an RGB image and apply Gaussian filter to add smoothing effect on it and display the results.
 

# '''

# import cv2 as cv

# filename = 'lab10task2.tif'

# img = cv.imread(filename, cv.IMREAD_COLOR)

# b,g,r = cv.split(img)

# b = cv.GaussianBlur(b, (5, 5), 0)
# g = cv.GaussianBlur(g, (5, 5), 0)
# r = cv.GaussianBlur(r, (5, 5), 0)

# img = cv.merge((b, g, r))

# cv.imshow("Gaussian Filter", img)
# cv.waitKey(0)



# '''
#     Task 03
#     Read an RGB image and apply sobel filters to see gradients colored images and display the results.

# Note: Apply filter on its each plane (Red, green and blue) one by one and combine the results. 
# '''


# import cv2 as cv

# filename = 'lab10task2.tif'

# img = cv.imread(filename, cv.IMREAD_COLOR)

# b,g,r = cv.split(img)

# b = cv.Sobel(b, cv.CV_64F, 1, 1, ksize=3)
# g = cv.Sobel(g, cv.CV_64F, 1, 1, ksize=3)
# r = cv.Sobel(r, cv.CV_64F, 1, 1, ksize=3)

# img = cv.merge((b, g, r))

# cv.imshow("Sobel Filter", img)
# cv.waitKey(0)


# '''
# Task 04
# Use the following figures to compute the FFT and display the results.
# '''
# import cv2 as cv
# import numpy as np

# filename1 = '1a.tif'
# filename2 = '1b.tif'

# img1 = cv.imread(filename1, cv.IMREAD_GRAYSCALE)
# img2 = cv.imread(filename2, cv.IMREAD_GRAYSCALE)

# f1 = np.fft.fftshift(np.fft.fft2(img1))
# f2 = np.fft.fftshift(np.fft.fft2(img2))

# magnitude1 = np.abs(f1)
# magnitude2 = np.abs(f2)

# magnitude1_log = np.log(1 + magnitude1)
# magnitude2_log = np.log(1 + magnitude2)

# # Normalize the magnitude spectrum images to uint8 format
# magnitude1_log_normalized = cv.normalize(magnitude1_log, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
# magnitude2_log_normalized = cv.normalize(magnitude2_log, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

# # Display magnitude spectrum images using OpenCV
# cv.imshow("Magnitude Spectrum of Image 1", magnitude1_log_normalized)
# cv.imshow("Magnitude Spectrum of Image 2", magnitude2_log_normalized)
# cv.waitKey(0)
# cv.destroyAllWindows()


