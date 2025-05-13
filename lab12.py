# """
# Task 1:
#          Calculate the Grey Level Co-occurrence Matrix (GLCM) for following images.
#              Then calculate the parameters of GLCM mentioned above for all the images and compare the results.
#
# """
#
# from skimage.feature import graycomatrix, graycoprops
# from matplotlib import  pyplot as plt
# import numpy as np
# import cv2 as cv
#
# #  reading image
# image = cv.imread("./lab12/Fig1128(c)(microporcessor-regular texture)-DO NOT SEND.tif", cv.IMREAD_GRAYSCALE)
#
# distances = [1]
# angles = [0, np.pi/4, np.pi/ 2, 3 * np.pi/4]
#
# glcm = graycomatrix(image, distances=distances, angles=angles)
# properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
#
# features = [graycoprops(glcm, prop) for prop in properties]
#
# plt.figure(figsize= (10, 5))
# plt.imshow(image)
# plt.title("Image")
# plt.show()
# for index in range(0, len(properties)):
#     print(properties[index], features[index])



# """
# Task 2:
#
#         Use spectral analysis to extract feature profile S(r) and S(theta) for given images and plot these feature
#         profiles.
#         Don't use built-in, create algorithm from scratch.
# """
#
# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
#
#
# def spectral_analysis(image):
#     #     create magnitude and phase matrix from fourier transform
#     f = np.fft.fft2(image)
#     fshift = np.fft.fftshift(f)
#
#     magnitude = np.abs(fshift)
#
#     phase = np.angle(fshift)
#
#     #     create s_r and s_theta
#     s_r = []
#     s_theta = []
#
#     for r in range(0, magnitude.shape[0]):
#         s_r.append(np.mean(magnitude[r]))
#
#     for theta in range(0, magnitude.shape[1]):
#         s_theta.append(np.mean(phase[theta]))
#
#     return s_r, s_theta
#
#
# image1 = cv.imread('./lab12/Fig1135(b)(ordered_matches).tif', cv.IMREAD_GRAYSCALE)
# image2 = cv.imread('./lab12/Fig1135(a)(random_matches).tif', cv.IMREAD_GRAYSCALE)
#
# s_r1, s_theta1 = spectral_analysis(image1)
# s_r2, s_theta2 = spectral_analysis(image2)
#
# # plot all the feature profiles in one figure but separately
# plt.figure(figsize=(10, 5))
# plt.subplot(2, 2, 1)
# plt.plot(s_r1)
# plt.title('S_r for image 1')
# plt.xlabel('r')
# plt.ylabel('S_r')
#
# plt.subplot(2, 2, 2)
# plt.plot(s_theta1)
# plt.title('S_theta for image 1')
# plt.xlabel('theta')
# plt.ylabel('S_theta')
#
# plt.subplot(2, 2, 3)
# plt.plot(s_r2)
# plt.title('S_r for image 2')
# plt.xlabel('r')
# plt.ylabel('S_r')
#
# plt.subplot(2, 2, 4)
# plt.plot(s_theta2)
# plt.title('S_theta for image 2')
# plt.xlabel('theta')
# plt.ylabel('S_theta')
#
# plt.tight_layout()
# plt.show()




"""
Task 3:
        Apply Local Binary pattern algorithm for following image.
        ./lab12/a-The-original-Lena-image-.tif
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def local_binary_pattern(image):
    #   local binary pattern algorithm
    padded_image = np.pad(image,1 )
    lbp = np.zeros_like(padded_image)
    for i in range(1, padded_image.shape[0] - 1):
        for j in range(1, padded_image.shape[1] - 1):
            center = padded_image[i, j]
            code = 0
            code |= (padded_image[i - 1, j - 1] > center) << 7
            code |= (padded_image[i - 1, j] > center) << 6
            code |= (padded_image[i - 1, j + 1] > center) << 5
            code |= (padded_image[i, j + 1] > center) << 4
            code |= (padded_image[i + 1, j + 1] > center) << 3
            code |= (padded_image[i + 1, j] > center) << 2
            code |= (padded_image[i + 1, j - 1] > center) << 1
            code |= (padded_image[i, j - 1] > center) << 0
            lbp[i, j] = code.astype(np.uint8)
    # Remove padding from lbp before returning
    lbp = lbp[1:-1, 1:-1]
    return lbp

image = cv.imread("./lab12/a-The-original-Lena-image-.tif", cv.IMREAD_GRAYSCALE)

lbp = local_binary_pattern(image)

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.imshow(image)
plt.title("Original lena Image")


plt.subplot(2, 1, 2)
plt.imshow(lbp)
plt.title("LBP of lena Image")
plt.tight_layout()
plt.show()