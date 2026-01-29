# """
# Task 1
# Use the following figures to compute the FFT and display the results.
# """
#
# import cv2 as cv
# import numpy as np
#
# filename1 = '1a.tif'
# filename2 = '1b.tif'
#
# img1 = cv.imread(filename1, cv.IMREAD_GRAYSCALE)
# img2 = cv.imread(filename2, cv.IMREAD_GRAYSCALE)
#
# f1 = np.fft.fftshift(np.fft.fft2(img1))
# f2 = np.fft.fftshift(np.fft.fft2(img2))
#
# magnitude1 = np.abs(f1)
# magnitude2 = np.abs(f2)
#
# magnitude1_log = np.log(1 + magnitude1)
# magnitude2_log = np.log(1 + magnitude2)
#
# # Normalize the magnitude spectrum images to uint8 format
# magnitude1_log_normalized = cv.normalize(magnitude1_log, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
# magnitude2_log_normalized = cv.normalize(magnitude2_log, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
#
# # Display magnitude spectrum images using OpenCV
# cv.imshow("Magnitude Spectrum of Image 1", magnitude1_log_normalized)
# cv.imshow("Magnitude Spectrum of Image 2", magnitude2_log_normalized)
# cv.waitKey(0)
# cv.destroyAllWindows()


# """
# Task 02
#  Apply smoothing effect on the given image (Fig 3) using Fourier transform technique. You will apply rectangular
#  shape low pass filter with cut off frequency 30 on given image. Display input image, its magnitude spectrum
#  and output image.
# """
#
# import cv2 as cv
# import numpy as np
#
# img = cv.imread("2.tif", cv.IMREAD_GRAYSCALE)
# f = np.fft.fftshift(np.fft.fft2(img))
#
# # Create a rectangular low pass filter
# rows, cols = img.shape
# cutoff_frequency = 30
# low_pass_filter = np.zeros((rows, cols), dtype=np.uint8)
# low_pass_filter[int(rows / 2) - cutoff_frequency:int(rows / 2) + cutoff_frequency,
# int(cols / 2) - cutoff_frequency:int(cols / 2) + cutoff_frequency] = 1
#
# # Apply the filter to the frequency domain representation of the image
# filtered_f = f * low_pass_filter
#
# # Compute the inverse Fourier transform to get the filtered image
# filtered_img = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_f)))
#
# # Normalize the filtered image to uint8 format
# filtered_img_normalized = cv.normalize(filtered_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
#
# # Display the input image, magnitude spectrum, and filtered image
# cv.imshow("Input Image", img)
# cv.imshow("Magnitude Spectrum", np.uint8(np.abs(10 * np.log(f))))
# cv.imshow("Filtered Image", filtered_img_normalized)
# cv.waitKey(0)
# cv.destroyAllWindows()

# """
# Task 03
# Find magnitude gradient of image by applying high pass filter of rectangular shape with cut off frequency 30.
# Display input image its magnitude spectrum and magnitude gradient.

# """

# import cv2 as cv
# import numpy as np

# img = cv.imread("3.tif", cv.IMREAD_GRAYSCALE)

# f = np.fft.fftshift(np.fft.fft2(img))

# # Create a rectangular high pass filter
# rows, cols = img.shape
# cutoff_frequency = 30
# high_pass_filter = np.ones((rows, cols), dtype=np.uint8)
# high_pass_filter[int(rows / 2) - cutoff_frequency:int(rows / 2) + cutoff_frequency,
#                     int(cols / 2) - cutoff_frequency:int(cols / 2) + cutoff_frequency] = 0

# # Apply the filter to the frequency domain representation of the image
# filtered_f = f * high_pass_filter

# # Compute the inverse Fourier transform to get the filtered image

# filtered_img = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_f)))

# # Normalize the filtered image to uint8 format
# filtered_img_normalized = cv.normalize(filtered_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

# # Display the input image, magnitude spectrum, and filtered image
# cv.imshow("Input Image", img)
# cv.imshow("Magnitude Spectrum", np.uint8(np.abs(7 * np.log(f))))
# cv.imshow("Magnitude Gradient", filtered_img_normalized)
# cv.waitKey(0)
# cv.destroyAllWindows()


"""
Task 04

Design Bandstop filter of width 30 and cutoff frequency 25 for noise removal in image. 
Bandstop filter: W is width of band, D is the distance D(u,v) from center of the filter, Do is cutoff    frequency. 
H(u,v)= {0         if Do ┤-(W )/2 ≤ D ≤ Do  + w/2   ;   1   Otherwise 

"""

import cv2 as cv
import numpy as np

img = cv.imread("4.tif", cv.IMREAD_GRAYSCALE)

f = np.fft.fftshift(np.fft.fft2(img))

# Create a bandstop filter
rows, cols = img.shape
width = 30
cutoff_frequency = 25
bandstop_filter = np.ones((rows, cols), dtype=np.uint8)
d = np.zeros((rows, cols), dtype=np.float32)
for i in range(rows):
    for j in range(cols):
        d[i, j] = np.sqrt((i - rows / 2) ** 2 + (j - cols / 2) ** 2)
        if cutoff_frequency - width / 2 <= d[i, j] <= cutoff_frequency + width / 2:
            bandstop_filter[i, j] = 0
                    

# Apply the filter to the frequency domain representation of the image
filtered_f = f * bandstop_filter

# Compute the inverse Fourier transform to get the filtered image
filtered_img = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_f)))

# Normalize the filtered image to uint8 format
filtered_img_normalized = cv.normalize(filtered_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

# Display the input image, magnitude spectrum, and filtered image

cv.imshow("Input Image", img)
cv.imshow("Magnitude Spectrum", np.uint8(np.abs(10 * np.log(f))))
cv.imshow("Filtered Image", filtered_img_normalized)
cv.waitKey(0)
cv.destroyAllWindows()

