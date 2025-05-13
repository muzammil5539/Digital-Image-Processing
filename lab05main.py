# # Task 01
#
# import cv2 as cv
# import numpy as np
# filename = 'Fig0241(a)(einstein low contrast).tif'
# image = cv.imread(filename, cv.IMREAD_GRAYSCALE)
# min=np.percentile(image, 5)
# max=np.percentile(image, 95)
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         if image[i,j] < min:
#             image[i, j] = 0
#         elif image[i, j] > max:
#             image[i, j] = 255
#         else:
#             image[i, j] = 255 * (image[i, j] - min)/(max-min)
# image = image.astype(np.uint8)
# cv.imshow('Contrast image', image)
# cv.waitKey()

# # Task 02
#
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
#
# filename = 'Fig0241(a)(einstein low contrast).tif'
# image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
# freq = [0] * 256
#
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         freq[image[i, j]] += 1
#
#
# freq2 = [0] * 256
#
# for i in range(len(freq2)):
#     freq2[i] = freq[i] / (image.shape[0] * image.shape[1])
#
# freq3 = [0] * 256
#
# for i in range(len(freq3)):
#     for j in range(i):
#         freq3[i] += freq2[j]
#
# freq4 = [0] * 256
# for i in range(len(freq3)):
#     freq4[i] = freq3[i] * 255
#
# image2 = np.zeros(image.shape, dtype=np.uint8)
#
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         image2[i, j] = freq4[image[i, j]]
# import cv2
# cv2.imshow('Enhanced Einstein', image2)
# cv2.waitKey()

# # Task 03 i global histogram equalization technique on the image
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# filename = 'Fig0326(a)(embedded_square_noisy_512).tif'
# image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
# freq = [0] * 256
#
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         freq[image[i, j]] += 1
#
# freq2 = [0] * 256
#
# for i in range(len(freq2)):
#     freq2[i] = freq[i] / (image.shape[0] * image.shape[1])
#
# freq3 = [0] * 256
#
# for i in range(len(freq3)):
#     for j in range(i):
#         freq3[i] += freq2[j]
#
# freq4 = [0] * 256
# for i in range(len(freq3)):
#     freq4[i] = freq3[i] * 255
#
# image2 = np.zeros(image.shape, dtype=np.uint8)
#
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         image2[i, j] = freq4[image[i, j]]
#
# cv2.imshow('Enhanced Image', image2)
# cv2.waitKey()

#
# # Task 03 ii local histogram equalization technique on the image
#
# import numpy as np
# import cv2
#
# def local_histogram_equalization(image, window_size):
#     image2 = np.zeros(image.shape, dtype=np.uint8)
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             window = image[max(0, i - window_size):min(image.shape[0], i + window_size),
#                      max(0, j - window_size):min(image.shape[1], j + window_size)]
#             freq = [0] * 256
#             for k in range(window.shape[0]):
#                 for l in range(window.shape[1]):
#                     freq[window[k, l]] += 1
#             freq2 = [0] * 256
#             for k in range(len(freq2)):
#                 freq2[k] = freq[k] / (window.shape[0] * window.shape[1])
#             freq3 = [0] * 256
#             for k in range(len(freq3)):
#                 for l in range(k):
#                     freq3[k] += freq2[l]
#             freq4 = [0] * 256
#             for k in range(len(freq3)):
#                 freq4[k] = freq3[k] * 255
#             image2[i, j] = freq4[image[i, j]]
#     return image2
#
#
# image = cv2.imread('Fig0326(a)(embedded_square_noisy_512).tif', cv2.IMREAD_GRAYSCALE)
# image2 = local_histogram_equalization(image, 3)
# cv2.imshow('Enhanced Image', image2)
# cv2.waitKey()

# # Task 04 (i)  global histogram equalization
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# filename = 'Fig0327(a)(tungsten_original).tif'
# image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
# freq = [0] * 256
#
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         freq[image[i, j]] += 1
#
# freq2 = [0] * 256
#
# for i in range(len(freq2)):
#     freq2[i] = freq[i] / (image.shape[0] * image.shape[1])
#
# freq3 = [0] * 256
#
# for i in range(len(freq3)):
#     for j in range(i):
#         freq3[i] += freq2[j]
#
# freq4 = [0] * 256
#
# for i in range(len(freq3)):
#     freq4[i] = freq3[i] * 255
#
# image2 = np.zeros(image.shape, dtype=np.uint8)
#
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         image2[i, j] = freq4[image[i, j]]
#
#
# cv2.imshow('Enhanced Image', image2)
# cv2.waitKey()

# Task 04 (ii) local histogram equalization statistics.

import numpy as np
import cv2

def local_histogram_equalization(image, window_size):
    image2 = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = image[max(0, i - window_size):min(image.shape[0], i + window_size),
                     max(0, j - window_size):min(image.shape[1], j + window_size)]
            freq = [0] * 256
            for k in range(window.shape[0]):
                for l in range(window.shape[1]):
                    freq[window[k, l]] += 1
            freq2 = [0] * 256
            for k in range(len(freq2)):
                freq2[k] = freq[k] / (window.shape[0] * window.shape[1])
            freq3 = [0] * 256
            for k in range(len(freq3)):
                for l in range(k):
                    freq3[k] += freq2[l]
            freq4 = [0] * 256
            for k in range(len(freq3)):
                freq4[k] = freq3[k] * 255
            image2[i, j] = freq4[image[i, j]]
    return image2


image = cv2.imread('Fig0327(a)(tungsten_original).tif', cv2.IMREAD_GRAYSCALE)

image2 = local_histogram_equalization(image, 3)
cv2.imshow('Enhanced Image', image2)
cv2.imwrite('enhanced_image.png', image2)
cv2.waitKey()



