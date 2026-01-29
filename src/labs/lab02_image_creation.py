import cv2 as cv
import numpy as np





# # Task 01
# def create_508_508_image(horizontal_margin,vertical_margin):
#     black_white = np.zeros((508,508),dtype='uint8')

#     black_white[horizontal_margin:508-horizontal_margin,vertical_margin:508-vertical_margin] = 255
#     cv.imshow('508 * 508 Image', black_white)
#     cv.waitKey()

# create_508_508_image(45,30)

# Task 02

width, height = 500, 300

gradient = np.zeros((height, width), dtype=np.uint8)
gradient_16 = np.zeros((height, width), dtype=np.uint8)
gradient_4= np.zeros((height, width), dtype=np.uint8)
gradient_1 = np.zeros((height, width), dtype=np.uint8)

for x in range(width):
    intensity = int((x / width) * 255)
    gradient[:, x] = intensity


for i in range(0, width, int(width/16)):
    gradient_16[:, i:i+int(width/16)] = gradient[:, i][0]

for i in range(0, width, int(width/(4))):
    gradient_4[:, i:i+int(width/(4))] = gradient[:, i][0]

for i in range(0, width, int(width/2)):
    gradient_1[:, i:i+int(width/2)] = gradient[:, i][0]

# Display the gradient
cv.imshow('Gradient', gradient)
cv.imshow('Gradient_16', gradient_16)
cv.imshow('Gradient_4', gradient_4)
cv.imshow('Gradient_1', gradient_1)
cv.waitKey(0)
cv.destroyAllWindows()

# # Task 03

# def coloured_image_at_border(horizontal_matrix_dimension,vertical_matrix_dimension):
#     couloured_image = np.zeros((horizontal_matrix_dimension,vertical_matrix_dimension,3),dtype='uint8')
#     couloured_image[:,:,:] = [255,255,255]
#     couloured_image[0:round(horizontal_matrix_dimension*(1/8))-1,0:round(vertical_matrix_dimension*(1/8))-1,:] = [0,0,255]
#     couloured_image[horizontal_matrix_dimension-round(horizontal_matrix_dimension*(1/8)):,0:round(vertical_matrix_dimension*(1/8))-1,:] = [0,255,0]
#     couloured_image[0:round(horizontal_matrix_dimension*(1/8))-1,vertical_matrix_dimension - round((1/8) * vertical_matrix_dimension) - 1:,:] = [255,0,0]
#     couloured_image[horizontal_matrix_dimension-round(horizontal_matrix_dimension*(1/8))-1:,vertical_matrix_dimension - round((1/8) * vertical_matrix_dimension) - 1:,:] = [0,0,0]
#     cv.imshow('My Image', couloured_image)
#     cv.waitKey()
    
# coloured_image_at_border(512,512)

# # Task 04
# filename = 'image2.jpg'
# colored_image = cv.imread(filename,0)
# flip_horizontal = cv.flip(colored_image,1)
# flip_vertical = cv.flip(colored_image,0)
# flip_both = cv.flip(colored_image,-1)
# image1 = np.concatenate((colored_image,flip_horizontal),0)
# image2= np.concatenate((flip_vertical,flip_both),0)
# image3 = np.concatenate((image1,image2),1)
# cv.imshow('Flipped Images',image3)
# cv.waitKey()


# # Task 05

# def eucledian_distance_image(a,b):
#      return round(np.sqrt(np.square(a-250) + np.square(b-250) ))

# def manhatten_distance_image(a,b):
#      return np.abs(a-250)+np.abs(b-250)

# def chessboard_distance_image(a,b):
#     return max(abs(a-250),abs(b-250))
    
# grey_scaleImage = np.zeros((501,501),dtype = 'uint8')
# for i in range(501):
#     for j in range(501):
#         grey_scaleImage[i][j] =   manhatten_distance_image(i,j)
# cv.imshow('eucledian_distance_image',grey_scaleImage)
# cv.waitKey()




# Rough Work

# coloured_image = np.ones((512,512,3),dtype='uint8')
# coloured_image[:,:,:] = [144,143,210]
# coloured_image[200:300,200:300,:] = [0,0,255]
# cv.imshow(winname='My Image',mat=coloured_image)
# cv.waitKey()