# #Task 01
# import numpy as np
# import cv2 as cv

# # Function to detect number of objects

# def num_of_objects(colored_image):
#   # Number of rows and coloumns
#   row, column = np.shape(colored_image)
  
#   # New Matrix to store labels
#   newMatrix = np.zeros((row, column), dtype=np.uint16) 
#   next_label = 1 

#   # Checking connectivity
#   for i in range(row):
#     for j in range(column):
#       if colored_image[i, j] == 1:
#         neighbors = []  # List for neighbours
#         if i > 0 and colored_image[i - 1, j] == 1:
#           neighbors.append(newMatrix[i - 1, j])
#         if j > 0 and colored_image[i, j - 1] == 1:
#           neighbors.append(newMatrix[i, j - 1])

#         if neighbors: 
#           newMatrix[i, j] = min(neighbors) 

#           # Merge the labels
#           for neighbor in neighbors:
#             newMatrix[newMatrix == neighbor] = min(neighbors)
#         else:
#           newMatrix[i, j] = next_label # Assign new label
#           next_label += 1 # Increment label for next object

#   # Count the number of unique labels (excluding label 0)
#   unique_labels = np.unique(newMatrix)
#   num_of_objects = len(unique_labels) - 1 
#   return num_of_objects


# my_image = cv.imread("cca.jpg",0)
# objects = num_of_objects(my_image)
# print(f'Number of Objects: {objects}')




import numpy as np

def calculate_distance(p1, p2, choice):
  if choice == 1:
    # Euclidean distance
    return np.sqrt(np.sum(np.square(np.array(p1) - np.array(p2))))  
  elif choice == 2:
    # Manhattan distance
    return np.sum(np.abs(np.array(p1) - np.array(p2)))
  elif choice == 3:
    # Chessboard distance
    return np.max(np.abs(np.array(p1) - np.array(p2)))
  else:
    raise ValueError("Invalid choice of distance metric. Please choose 1 (Euclidean), 2 (Manhattan), or 3 (Chessboard).")

# Example usage
p1 = (10, 20)
p2 = (15, 25)

euclidean_distance = calculate_distance(p1, p2, choice=1)
manhattan_distance = calculate_distance(p1, p2, choice=2)
chessboard_distance = calculate_distance(p1, p2, choice=3)

print(f"Euclidean distance: {euclidean_distance}")
print(f"Manhattan distance: {manhattan_distance}")
print(f"Chessboard distance: {chessboard_distance}")
