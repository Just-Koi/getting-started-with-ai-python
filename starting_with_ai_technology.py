# Add a function that tells on which line of the code is executed a line of it

# ===============================================================
# ===============================================================
# ===============================================================
#         ++BEGGINERS PROBLEMS THAT i HAVE MET WITH++

# - Don't name a file with title that you want to import. Sounds stupid? 
# Well, many people don't know that. Don't name a file "numpy.py" while 
# you're importing numpy, name it differently, for example: numpbyBegginers.py

# - You need an interpreter in order to import libarires. When you 
# have installed an libary, import it and it doesn't show up as installed
# then it means that you firstly need to select an interpreter for your 
# python. For example I'm using conda interpreter. 

# ===============================================================
# ===============================================================
# ===============================================================

def addLine(a, b):
    string_a = str(a)
    string_b = str(b)
    return "From line " + string_a + " to line " + string_b + ": "

import numpy as np

# NumPy provides a powerful data structure called an array, 
# which is similar to a list but allows for efficient numerical
# operations. You can create an array using the np.array() 
# function, passing in a list or a nested list as an argument:

my_list = [1, 2, 3, 4, 5]
my_array = np.array(my_list)
print(addLine(33, 34), my_array) # Output: [1, 2, 3, 4, 5]
#print("from line 14 to line 15: ", my_array) 

# Once you have created an array, you can perform various
# operations on it. Here are a few examples:

# Accessing elements
print(addLine(42, 42), my_array[0])  # Output: 1
print(addLine(43, 43), my_array[2])  # Output: 3

# Slicing
print(addLine(46, 46), my_array[1:4])  # Output: [2, 3, 4]

# Arithmetic operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a + b  # Element-wise addition
print(addLine(49, 51), c)  # Output: [5, 7, 9]

# Universal functions
d = np.sqrt(a)  # Square root of each element
print(addLine(55, 55), d)  # Output: [1.0, 1.414, 1.732]

# ========================================================
# ========================================================
# ========================================================

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

print(addLine(70, 72), result)  # Output: 4.0

# ========================================================
# ========================================================
# ========================================================

# NumPy arrays have several attributes and methods that 
# provide useful information or perform operations on
# the array. Here are a few examples:

# Shape
print(addLine(85, 85), my_array.shape)  # Output: (5,) - a 1-dimensional array with 5 elements

# Size
print(addLine(88, 88), my_array.size)  # Output: 5 - the total number of elements in the array

# Reshape
reshaped_array = my_array.reshape((5, 1))
print(addLine(91, 92), reshaped_array)  # Output: [[1], [2], [3], [4], [5]]

# Mean
mean_value = my_array.mean()
print(addLine(95, 96), mean_value)  # Output: 3.0 - the average of the array elements


# These are just some basic operations and methods in NumPy. NumPy
# provides many more functionalities for numerical computing, including
# advanced indexing, array manipulation, linear algebra, and more. You
# can refer to the NumPy documentation for a comprehensive list of features
# and functions: https://numpy.org/doc/

# ========================================================
# ========================================================
# ========================================================

import scipy

# SciPy is a powerful open-source library for scientific and
# technical computing in Python. It provides a collection of
# mathematical algorithms and functions built on top of the
# NumPy library, extending its capabilities to solve a wide
# range of scientific and engineering problems.

# SciPy provides a wide range of scientific computing functions. 
# Here are a few examples of what you can do with SciPy:

    # 1. Integration:
    #     You can use the scipy.integrate module to perform integration.
    #     For example, let's integrate a function using the quad() function:

from scipy.integrate import quad

result, error = quad(lambda x: x**2, 0, 2)
print(addLine(126, 127), result)  # Output: 2.666666666666667

# ========================================================
# ========================================================
# ========================================================

# What is a quad function?

# In the context of scientific computing and numerical
# integration, the "quad" function refers to a specific
# function called quad() provided by the SciPy library 
# in Python.

# The quad() function is used for numerical integration, 
# specifically for computing a definite integral of a function
# over a specified interval. It provides an efficient and accurate
# way to approximate the integral using various numerical integration 
# techniques.

# Here's the basic syntax of the quad() function:
def func(c):
    return c
result, error = quad(func, a, b)

    # - `func` is the function to be integrated.
    # - `a` and `b` are the lower and upper limits of the integration interval, respectively.
    # - `result` is the estimated value of the integral.
    # - `error` is an estimate of the absolute error in the result.

# The quad() function takes a callable function func
# and returns an estimate of the integral of func over the interval
# [a, b]. The function func should be defined and properly implemented
# to represent the integrand.

# Here's a simple example to demonstrate the usage of quad():

def integrand(x):
    return x**2

result, error = quad(integrand, 0, 1)
print(addLine(161, 165), result)  # Output: 0.33333333333333337

# In this example, the integrand() function defines the function
# x^2 to be integrated. The quad() function is then called with 
# integrand as the function to integrate and the interval [0, 1]. 
# The resulting estimated integral value is assigned to result 
# and printed.

# The quad() function in SciPy is a versatile tool for numerical integration 
# and allows you to handle a wide range of integrals efficiently. It supports 
# both simple and complex integrands and provides different integration techniques 
# to achieve accurate results.

# ========================================================
# ========================================================
# ========================================================