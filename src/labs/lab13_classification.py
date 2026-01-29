# Task 01:Task 1: Using the dataset.csv file as input data,
# write the code for minimum distance classification.
# Make a confusion matrix and calculate precision and accuracy parameters.

import numpy as np
import pandas as pd

# Read the dataset
df = pd.read_csv('Iris.csv')

# Drop the 'Id' column if it exists
if 'Id' in df.columns:
  df.drop('Id', axis=1, inplace=True)

# Split the dataset into training and testing
train_df = pd.concat([df.iloc[:40], df.iloc[50:90]])
test_df = pd.concat([df.iloc[40:50], df.iloc[90:]])


train_x = (train_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
test_x = (test_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])


groups = train_df.groupby('Species')

means = groups.mean()


# Assign the class with minimum distance to each test sample
predicted_y = []
for index, row in test_x.iterrows():
    distances = np.linalg.norm(row - means, axis=1)
    predicted_y.append(groups.get_group(means.index[np.argmin(distances)])['Species'].values[0])

# Calculate the confusion matrix
confusion_matrix = pd.crosstab(test_df['Species'], predicted_y, rownames=['Actual'], colnames=['Predicted'])

# Calculate precision, recall, and accuracy
precision = np.diag(confusion_matrix.values) / np.sum(confusion_matrix.values, axis=1)
recall = np.diag(confusion_matrix.values) / np.sum(confusion_matrix.values, axis=0)
accuracy = np.sum(np.diag(confusion_matrix.values)) / np.sum(np.sum(confusion_matrix.values))

print('Precision:', precision)
print('Recall:', recall)
print('Accuracy:', accuracy)

print('Confusion Matrix:')
print(confusion_matrix)

# #  Task 02
# import numpy as np
# import pandas as pd
#
#
# # Read the dataset
# def read_inputdata(file):
#     data_frame = pd.read_csv(file)
#     data_frame.drop('Id', axis=1, inplace=True)
#     return data_frame
#
#
# def calculate_distance(instance1, instance2):
#     return ((instance1['SepalLengthCm'] - instance2['SepalLengthCm']) ** 2 + (
#                 instance1['SepalWidthCm'] - instance2['SepalWidthCm']) ** 2 + (
#                         instance1['PetalLengthCm'] - instance2['PetalLengthCm']) ** 2 + (
#                         instance1['PetalWidthCm'] - instance2['PetalWidthCm']) ** 2) ** 0.5
#
#
# def find_neighbours(k, train_x, train_y, test_instance):
#     distances = []
#     for index, row in train_x.iterrows():
#         distances.append(calculate_distance(row, test_instance))
#     return np.argsort(distances)[:k], train_y.iloc[np.argsort(distances)[:k]]
#
# def split_dataset(df):
#     train_df = pd.concat([df.iloc[:40], df.iloc[50:90]])
#     test_df = pd.concat([df.iloc[40:50], df.iloc[90:100]])
#
#     train_x = train_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
#     test_x = test_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
#     train_y = train_df['Species']  # Retain the 'Species' column for the training set
#     test_y = test_df['Species']    # Retain the 'Species' column for the test set
#
#     return train_x, test_x, train_y, test_y
#
#
# def confusion_matrix(test_y, predicted_y):
#     return pd.crosstab(test_y, predicted_y, rownames=['Actual'], colnames=['Predicted'])
#
#
# def precision_recall_accuracy(confusion_matrix):
#     precision = np.diag(confusion_matrix.values) / np.sum(confusion_matrix.values, axis=1)
#     recall = np.diag(confusion_matrix.values) / np.sum(confusion_matrix.values, axis=0)
#     accuracy = np.sum(np.diag(confusion_matrix.values)) / np.sum(np.sum(confusion_matrix.values))
#
#     return precision, recall, accuracy
#
#
# df = read_inputdata('Iris.csv')
#
# train_x, test_x, train_y, test_y = split_dataset(df)
#
# predicted_y = []
# k = 3
#
#
# def get_response(y):
#     return y.value_counts().idxmax()
#
# b = 0
# for index, row in test_x.iterrows():
#     nearest_neighbour_array, y = find_neighbours(k, train_x, train_y, row)
#     predicted_y.append(get_response(y))
#
#
# confusion_matrix = confusion_matrix(test_y, predicted_y)
# precision, recall, accuracy = precision_recall_accuracy(confusion_matrix)
#
# print('Precision:', precision)
# print('Recall:', recall)
# print('Accuracy:', accuracy)
#
# print('Confusion Matrix:')
# print(confusion_matrix)
