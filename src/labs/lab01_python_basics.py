# # Task 01

# def sort_tuple(unsorted_tuple):
#     list1 = list(unsorted_tuple)
    
#     # Bubble sort algorithm
#     for i in range(len(list1)):
#         for j in range(len(list1) - i - 1):
#             if list1[j][1] > list1[j + 1][1]:
#                 list1[j], list1[j + 1] = list1[j + 1], list1[j]
#     return tuple(list1)

# tuple1 = (('a',23),('b',37),('c',11),('d',29))
# print(sort_tuple(tuple1))


   
# # Task 02

# def minimum_value(dict):
#     minimum = 1000
#     number = list(dict.values())
#     keys = list(dict.keys())
#     for key in dict.keys():
#         if(minimum> dict[key]):
#             minimum = dict[key]
#     return (keys[number.index(minimum)])
# Dict = {'Physics': 82, 'Math': 65,'history': 75}
# print (minimum_value(Dict))

# # Task 03
# def number_System(num, sys):
#     sys = str(sys).upper()
#     divisor = 0
#     if sys == 'O':
#         divisor = 8
#     elif sys == 'H':
#         divisor = 16
#     elif sys == 'B':
#         divisor = 2
#     converted_number = ''
#     while num >= divisor:
#         remainder = num % divisor
#         if 10 <= remainder <= 15:
#             # Convert the numbers from 10 to 15 into 'A' to 'F'
#             converted_number = chr(ord('A') + remainder - 10) + converted_number
#         else:
#             converted_number = str(remainder) + converted_number
#         num = num // divisor
#     return str(num) + converted_number if num else converted_number

# num = int(input('Enter Number: '))
# sys = input('Specify Number System to convert: Octal [O], Hexadecimal [H], Binary [B] ')
# print(number_System(num, sys))


# # Task 04
# def min_max_Normalization(numbers):
#    numbers1 = []
#    for num in numbers:
#       numbers1.append(num)
#    numbers1.sort()
#    min = numbers1[0]
#    max = numbers1[len(numbers1)-1]
#    for i in range(0 ,len(numbers)):
#       numbers[i] = (numbers[i]-min)/(max-min)
#    return numbers
   
# list1 = [5,7,8,10,6,4]
# print(min_max_Normalization(list1))


# # Task 05
# def average_temperatures(temperatures):


#   if len(temperatures) != 6 or len(temperatures[0]) != 6:
#     print("Input must be a 6x6 array")
#     return

#   average_temps = []

#   for i in range(0, 6, 2):
#     for j in range(0, 6, 2):
#       grid_sum = temperatures[i][j] + temperatures[i][j + 1] + temperatures[i + 1][j] + temperatures[i + 1][j + 1]
#       average_temp = grid_sum // 4
#       average_temps.append(average_temp)

#   return average_temps

# temperatures = [
#   [1, 2, 3, 4, 5, 6],
#   [10, 11, 12, 13, 14, 15],
#   [19, 20, 21, 22, 23, 24],
#   [28, 29, 30, 31, 32, 33],
#   [37, 38, 39, 40, 41, 42],
#   [46, 47, 48, 49, 50, 51],
# ]

# average_temps = average_temperatures(temperatures)

# print(average_temps)


## Home Tasks

# # Task 1

# def add_lists(list1, list2):
#    result = []
#    max_len = max(len(list1), len(list2))    
#    for i in range(max_len):
#         if i < len(list1) and i < len(list2):
#             result.append(list1[i] + list2[i])
#         elif i < len(list1):
#             result.append(list1[i])
#         else:
#             result.append(list2[i])
    
#    return result

# list1 = ["M", "na", "i", "Ke"]
# list2 = ["y", "me", "s", "lly"]
# print(add_lists(list1, list2))


# Task 02

def is_Palindrome(s):
    s = s.lower()    
    return s == s[::-1]

print(f'Palindrome\nRadar: {is_Palindrome("radar")}')  # True
print(f'Palindrome\nCeramic: {is_Palindrome("ceramic")}')  # False
