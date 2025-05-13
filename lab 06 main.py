import cv2 as cv
import numpy as np


def create_mask(size):
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            mask[i, j] = 1 / (size * size)
    return mask


def add_image_padding(image, padding_size):
    size = (image.shape[0] + 2 * padding_size, image.shape[1] + 2 * padding_size)
    padded_image = np.zeros(size)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            padded_image[i + padding_size, j + padding_size] = image[i, j]
    return padded_image


def apply_filter(image, mask):
    mask_size = mask.shape
    image_size = image.shape
    filtered_image = np.zeros_like(image)

    for k in range(mask_size[0] // 2, image_size[0] - mask_size[0] // 2):
        for l in range(mask_size[1] // 2, image_size[1] - mask_size[1] // 2):
            value = 0
            for m in range(-mask_size[0] // 2, mask_size[0] // 2 + 1):
                for n in range(-mask_size[1] // 2, mask_size[1] // 2 + 1):
                    value += image[k + m, l + n] * mask[m + mask_size[0] // 2, n + mask_size[1] // 2]
            filtered_image[k, l] = value
    return filtered_image


def normalize_image(image):
    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    return image.astype(np.uint8)


size = 3
image = cv.imread('lab5task3.tif', cv.IMREAD_GRAYSCALE)
mask = create_mask(size)
padded_image = add_image_padding(image, size // 2)
filtered_image = apply_filter(padded_image, mask)
normalized_image = normalize_image(filtered_image)
cv.imwrite('filtered_image_lab_06003_3x3.png', normalized_image)

size = 5
mask = create_mask(size)
padded_image = add_image_padding(image, size // 2)
filtered_image = apply_filter(padded_image, mask)
normalized_image = normalize_image(filtered_image)
cv.imwrite('filtered_image_lab_06003_5x5.png', normalized_image)