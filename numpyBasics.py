# Add a function that tells on which line of the code is executed a line of it

def addLine(a, b):
    line = "From line ", a, " to line ", b, ": "
    return line

import numpy as np

# NumPy provides a powerful data structure called an array, 
# which is similar to a list but allows for efficient numerical
# operations. You can create an array using the np.array() 
# function, passing in a list or a nested list as an argument:

my_list = [1, 2, 3, 4, 5]
my_array = np.array(my_list)
print(addLine(14, 15), my_array) # Output: [1, 2, 3, 4, 5]
#print("from line 14 to line 15: ", my_array) 

# Once you have created an array, you can perform various
# operations on it. Here are a few examples:

# Accessing elements
print(my_array[0])  # Output: 1
print(my_array[2])  # Output: 3

# Slicing
print(my_array[1:4])  # Output: [2, 3, 4]

# Arithmetic operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a + b  # Element-wise addition
print(c)  # Output: [5, 7, 9]

# Universal functions
d = np.sqrt(a)  # Square root of each element
print(d)  # Output: [1.0, 1.414, 1.732]

# Good to know: What is 'sqrt' ?
# - sqrt is a mathematical function that calculates the 
# square root of a given number. It is an abbreviation 
# for "square root."
# - In Python, you can use the sqrt function from the 
# math module to calculate the square root. 
# Here's an example:

import math
x = 16
result = math.sqrt(x)

print(result)  # Output: 4.0