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

    # def func(c):
    #     return c
    # result, error = quad(func, a, b)

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

# 2. Optimization:
# You can use the scipy.optimize module to solve optimization
# problems. Here's an example using the minimize() function 
# to find the minimum of a function:

from scipy.optimize import minimize

def func(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

initial_guess = [0, 0]
result = minimize(func, initial_guess)
print(addLine(192, 196), result.x)  # Output: [1. 2.5]

# 3. Linear Algebra:
# You can use the scipy.linalg module for linear algebra operations. 
# Here's an example to solve a system of linear equations:

from scipy import linalg

A = [[1, 2], [3, 4]]
b = [5, 6]
x = linalg.solve(A, b)
print(addLine(205, 207), x)  # Output: [-4.   4.5]

# ========================================================
# ========================================================
# ========================================================

# What is linear algebra?

# Linear algebra is a branch of mathematics that deals with
# the study of vectors, vector spaces, linear transformations, 
# and systems of linear equations. It provides a framework for 
# solving mathematical problems involving linear relationships 
# and structures.

# In linear algebra, the fundamental objects of study are vectors
# and matrices. Here are some key concepts in linear algebra:

#     - Vectors: Vectors are quantities that have both magnitude and direction. 
#     They can be represented as ordered lists of numbers or as column matrices. 
#     Vectors can be added together, multiplied by scalars, and used to represent 
#     points, directions, or physical quantities.

#     - Matrices: Matrices are rectangular arrays of numbers, organized into rows 
#     and columns. They are used to represent linear transformations and systems
#     of linear equations. Matrices can be added, multiplied, and manipulated 
#     using various operations.

#     - Linear Transformations: Linear transformations are functions that preserve
#     vector addition and scalar multiplication. They map vectors from one vector 
#     space to another while maintaining the properties of linearity. Examples of 
#     linear transformations include rotations, scaling, and reflections.

#     - Systems of Linear Equations: Systems of linear equations are sets of equations
#     involving multiple variables, where each equation represents a linear relationship. 
#     The goal is to find the values of the variables that satisfy all the equations 
#     simultaneously. Techniques such as Gaussian elimination and matrix inversion are 
#     commonly used to solve these systems.

#     - Eigenvectors and Eigenvalues: Eigenvectors and eigenvalues are associated with 
#     square matrices. Eigenvectors are special vectors that remain in the same direction,
#     up to a scalar factor, when multiplied by a matrix. Eigenvalues are the corresponding 
#     scalar factors. They have applications in various areas, including principal component 
#     analysis, image compression, and graph analysis.

# Linear algebra has numerous applications in various fields, including physics, engineering, 
# computer science, data analysis, and machine learning. It provides a powerful set of tools
# and techniques for solving problems involving linear relationships, transformations, 
# and systems of equations.

# In Python, the NumPy and SciPy libraries provide comprehensive support for linear algebra
# operations, making it convenient to perform calculations and manipulations involving 
# vectors and matrices.

# ========================================================
# ========================================================
# ========================================================

# 4. Interpolation:
# You can use the scipy.interpolate module for interpolation. 
# Here's an example of cubic spline interpolation:

from scipy.interpolate import CubicSpline

x = [0, 1, 2, 3, 4]
y = [0, 1, 4, 9, 16]

cs = CubicSpline(x, y)
interpolated_value = cs(2.5)
print(addLine(271, 275), interpolated_value)  # Output: 6.25

# ========================================================
# ========================================================
# ========================================================

# What is Cubic spline?

# Cubic spline interpolation is a technique used to construct a 
# smooth and continuous curve between given data points. It is 
# commonly used in data analysis, numerical approximation, and 
# computer graphics.

# The basic idea behind cubic spline interpolation is to fit a piecewise cubic 
# polynomial to the data points, ensuring that the resulting curve passes 
# through each point and exhibits smoothness properties. The piecewise nature 
# means that the curve is composed of individual cubic segments that join 
# together at the data points.

# To perform cubic spline interpolation, the following steps are typically involved:

    # - Input: Obtain a set of data points (x[i], y[i]) where x[i] represents the independent 
    # variable and y[i] represents the dependent variable.

    # - Knot Selection: Determine the interval (or "knot") for each data point. These intervals
    # define the regions where the cubic polynomials will be fitted.

    # - Polynomial Fitting: Construct a cubic polynomial within each interval, ensuring that the curve 
    # passes through the corresponding data points. This is achieved by solving a system of equations 
    # that impose the interpolation conditions.

    # - Smoothness Constraints: Apply smoothness constraints at the interior knot points to ensure 
    # continuity and smoothness of the curve. This typically involves enforcing continuity of the first 
    # and second derivatives across the intervals.

    # - Evaluate the Spline: Once the cubic spline is constructed, it can be used to estimate function 
    # values at any desired point within the interpolation range. The polynomial segments are combined to 
    # form a continuous curve that approximates the original data.

# Cubic spline interpolation provides a flexible and smooth representation of data, allowing for 
# interpolation and estimation of values between the given data points. It helps to overcome issues 
# such as overfitting or excessive oscillations that can occur with simpler interpolation methods.

# In Python, the SciPy library provides the scipy.interpolate module, which includes functions for 
# cubic spline interpolation, such as CubicSpline and interp1d. These functions can be used to
# perform cubic spline interpolation and obtain the interpolated values for a given set of data points.

# It's worth noting that there are other interpolation techniques available, such as linear 
# interpolation and polynomial interpolation, each with its own characteristics and use cases. 
# The choice of interpolation method depends on the specific requirements and nature of the 
# data being analyzed.

# ========================================================
# ========================================================
# ========================================================

# These are just a few examples of what you can do with SciPy. 
# It provides many more submodules and functions for various 
# scientific computing tasks, including signal processing, 
# image processing, statistics, and more. You can refer to the 
# SciPy documentation for a comprehensive list of features 
# and functions: https://docs.scipy.org/doc/scipy/reference/

# ========================================================
# ========================================================
# ========================================================

import matplotlib.pyplot as plt

# Matplotlib is a popular Python library used for creating static,
# animated, and interactive visualizations in Python. It provides 
# a comprehensive set of functions and classes for generating a 
# wide range of plots, charts, and graphs.

# Key features of Matplotlib include:

# 1. Plotting Functions: Matplotlib provides a variety of plotting
#    functions to create different types of plots, such as line plots, 
#    scatter plots, bar plots, histogram plots, pie charts, and more.

# 2. Customization Options: Matplotlib allows extensive customization 
#    of plot elements such as axes, labels, titles, colors, line styles, 
#    markers, and legends. This flexibility enables you to create plots with
#    a high level of control over their appearance.

# 3. Support for Multiple Backends: Matplotlib supports various rendering
#    backends, including different graphical toolkits and file formats. 
#    This allows you to generate plots in different environments, such 
#    as interactive GUI windows, Jupyter notebooks, or save them as image files.

# 4. Integration with NumPy and Pandas: Matplotlib seamlessly integrates with 
#    other scientific computing libraries like NumPy and Pandas, allowing you 
#    to plot data directly from these libraries' data structures.

# 5. Matplotlib Gallery and Examples: Matplotlib provides a vast gallery of 
#    example plots, covering a wide range of use cases and visualizations. These
#    examples serve as a useful resource for learning and understanding the 
#    capabilities of the library.

# ==Data==
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]

# ==Plotting==
# plt.plot(x, y, 'ro-')  # 'ro-' specifies red circles with solid lines
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Line Plot')
# plt.grid(True)

# ==Displaying the plot==
# plt.show()

# In this example, the plot() function is used to create a line plot
# by providing x and y values. Additional functions like xlabel(), 
# ylabel(), title(), and grid() are used to add labels, a title, and a 
# grid to the plot. Finally, show() is called to display the plot.

# Creating Basic Plots:
# Matplotlib provides various functions to create different 
# types of plots. Here are a few examples:

# 1. Line Plot:

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('From line 400 to line 406: Line Plot')
plt.show()

# 2. Scatter Plot:

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.scatter(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('From line 410 to line 416: Scatter Plot')
plt.show()

# 3. Bar Plot:

x = ['A', 'B', 'C', 'D']
y = [10, 5, 7, 12]
plt.bar(x, y)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('From line 420 to line 426: Bar Plot')
plt.show()

# Cutomiznig plots:
# Matplotlib allows you to customize various aspects of 
# your plots. Here are a few common customizations:

# 1. Adding Legends:

x = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]
y2 = [1, 3, 5, 7, 9]
plt.plot(x, y1, label='Line 1')
plt.plot(x, y2, label='Line 2')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('From line 434 to line 443: Multiple Lines')
plt.legend()
plt.show()

# 2. Changing Line Styles and Marker Types:

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y, linestyle='--', marker='o', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('From line 447 to line 453: Customized Line Plot')
plt.show()

# 3. Adding Annotations:

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('From line 457 to line 464: Line Plot with Annotation')
plt.annotate('Point of Interest', xy=(3, 6), xytext=(4, 8), arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.show()

# Saving plots:
# You can save your plots as image files using the savefig()
# function. Here's an example:

# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]
# plt.plot(x, y)
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Line Plot')
# plt.savefig('line_plot.png')

# These are just some basic operations and customizations in Matplotlib. 
# Matplotlib provides many more functionalities for creating complex 
# visualizations, including histograms, pie charts, 3D plots, and more. 
# You can refer to the Matplotlib documentation for a comprehensive list 
# of features and functions: https://matplotlib.org/stable/contents.html

# ========================================================
# ========================================================
# ========================================================

import IPython as IPython

# IPython, short for "Interactive Python," is an enhanced interactive 
# shell for executing Python code. It provides an interactive environment 
# that offers features and improvements over the standard Python shell, 
# making it a popular choice among Python developers and data scientists.

# Some key features of IPython include:

#   - Enhanced Interactive Shell: 
#     IPython provides an interactive shell with features like 
#     tab-completion, syntax highlighting, history management 
#     (command history and input/output caching), and easy access 
#     to system shell commands.

#   - Rich Display Capabilities: 
#     IPython supports rich media representations, allowing you 
#     to display images, videos, audio, and HTML content directly 
#     within the shell. It also supports rendering of plots and 
#     visualizations using libraries like Matplotlib.

#   - Command-Line Magic Commands: 
#     IPython introduces special commands called "magic commands" 
#     that provide additional functionality for code execution, 
#     system interaction, debugging, and profiling. Magic commands start 
#     with a percent sign (%) or two percent signs (%%) and can be used to
#     perform various tasks like timing code execution, loading files, 
#     running system commands, and more.

#     %run my_script.py - Runs an external Python script

#   - Command History:
#     IPython keeps a history of your commands, allowing you to navigate 
#     and reuse previous commands. You can access your command history 
#     using the up and down arrow keys.

#   - Inline Help:
#     You can get help on Python objects and functions directly within IPython. 
#     Use a question mark (?) to display a help message. For example:

#     help(len)  - Displays help for the len() function

#   - Jupyter Notebook Integration: 
#     IPython serves as the underlying kernel for Jupyter Notebook, which 
#     is a web-based interactive computational environment. Jupyter Notebook 
#     allows you to create and share documents containing live code, equations, 
#     visualizations, and narrative text.

#   - Interactive Data Analysis and Exploration: 
#     IPython provides an interactive environment suitable for data analysis and
#     exploration. It integrates well with popular data analysis libraries like NumPy,
#     Pandas, Matplotlib, and SciPy, allowing you to work with data, perform computations, 
#     and visualize results seamlessly.

print("To start an IPython session, you can simply open a terminal\nor command prompt and run the command ipython. This will\nlaunch the IPython shell, where you can enter Python code interactively\nand take advantage of its features. After typing in ipython\nin terminal, try to type in: '%'run my_script.py -> without quotes.\nTo exit IPython, you can type exit, quit(), or press\nCtrl+D. IPython provides many more features and capabilities for\ninteractive Python programming and data exploration. You can refer to\nthe IPython documentation for a comprehensive list of features and\nfunctionalities: https://ipython.readthedocs.io/")

# IPython provides a more interactive and user-friendly experience compared to the
# standard Python shell, making it a powerful tool for prototyping, experimentation,
# and interactive coding workflows.

# ========================================================
# ========================================================
# ========================================================

import sklearn

# Scikit-learn, often abbreviated as sklearn, is a popular 
# open-source machine learning library for Python. It provides a 
# wide range of tools and algorithms for various machine learning
# tasks, including classification, regression, clustering, 
# dimensionality reduction, model selection, and preprocessing.

# Key features of scikit-learn include:

# - Simple and Consistent API: 
#   Scikit-learn provides a unified and easy-to-use API, making 
#   it accessible for beginners while still offering flexibility 
#   and control for advanced users. The consistent interface across
#   different algorithms makes it easier to experiment and switch
#   between models.

# - Comprehensive Set of Algorithms: 
#   Scikit-learn includes a broad collection of machine learning 
#   algorithms. It covers both supervised learning (e.g., 
#   linear regression, logistic regression, support vector 
#   machines, decision trees, random forests, gradient boosting) 
#   and unsupervised learning (e.g., clustering, dimensionality 
#   reduction, anomaly detection).

# - Data Preprocessing and Feature Extraction: 
#   Scikit-learn offers various tools for preprocessing data, 
#   including handling missing values, scaling features, encoding 
#   categorical variables, and feature selection. These preprocessing 
#   techniques help in preparing the data for machine learning models.

# - Model Evaluation and Selection: 
#   Scikit-learn provides functions and metrics to evaluate and compare
#   models. It offers tools for cross-validation, hyperparameter tuning,
#   model selection, and performance evaluation metrics such as accuracy,
#   precision, recall, F1-score, and more.

# - Integration with NumPy and Pandas: 
#   Scikit-learn seamlessly integrates with other popular Python 
#   libraries such as NumPy and Pandas. This allows for efficient 
#   handling and manipulation of numerical arrays and tabular data, 
#   respectively.

# - Extensibility and Integration: 
#   Scikit-learn is designed to be easily extensible, 
#   allowing developers to implement their own algorithms 
#   and incorporate them into the library. It also integrates well 
#   with other machine learning and data science libraries, such as
#   TensorFlow, Keras, and PyTorch.

# Using scikit-learn for Machine Learning Tasks:

# - Data Preparation:
#  scikit-learn provides utilities for data preprocessing, 
#  such as handling missing values, scaling features, and encoding
#  categorical variables. Here's an example of scaling features 
#  using the StandardScaler:

from sklearn.preprocessing import StandardScaler
# Assuming X is your feature matrix
X = [[0,1,2],[1,2,0]]
print (addLine(613, 613), 'Feature matrix: ', X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print (addLine(610, 616), 'Feature matrix scaled: ', X_scaled)

# ========================================================
# ========================================================
# ========================================================

# What is a feature matrix?

# A feature matrix, also known as a feature array or feature dataset,
# is a structured representation of the features or variables used to
# describe the data in a machine learning problem. It is typically
# represented as a two-dimensional matrix or array, where each row
# corresponds to a data instance or observation, and each column 
# represents a specific feature or attribute.

# In a feature matrix, the rows correspond to individual samples or 
# data points, and the columns represent the different features or 
# variables associated with each sample. Each element of the matrix 
# holds the value of a particular feature for a specific data instance.

# The feature matrix is often denoted as X in machine learning notation, 
# where X represents the input variables or features. It is crucial to 
# preprocess and normalize the feature matrix before training a machine 
# learning model to ensure that all features are on a similar scale and 
# have meaningful representations.

# Note that in some cases, the feature matrix may also include a column 
# representing the target variable (the variable we aim to predict), 
# especially in supervised learning problems. However, the target variable 
# is typically kept separate from the feature matrix during training and 
# prediction.

# ========================================================
# ========================================================
# ========================================================

# - Model Training and Evaluation:
#   scikit-learn provides a variety of algorithms for classification, 
#   regression, clustering, and more. Here's an example of training a 
#   logistic regression classifier:

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
X = np.array([[1, 0, -1, 0], [-1, 1, 1, 0], [0, 1, -1, 1]])
y = np.array([1,0,1]) 
# Assuming X is your feature matrix and y is the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print (addLine(657, 671), 'Accuracy of the model is: ', accuracy)

y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print (addLine(658, 675), 'Accuracy of the train of the model is: ', train_accuracy)
print ('Model is accureate.')

# ========================================================
# ========================================================
# ========================================================

# What is LogisticRegression() ?

# Logistic Regression is a statistical algorithm used for binary 
# classification tasks, where the goal is to predict a binary outcome
# (e.g., yes/no, true/false, 0/1) based on a set of input features. 
# Despite its name, logistic regression is actually a classification algorithm
# rather than a regression algorithm.

# The logistic regression model uses the logistic function (
# also known as the sigmoid function) to model the relationship between the input 
# features and the probability of the binary outcome. The logistic function maps any
# real-valued number to a value between 0 and 1, which represents the probability 
# of the positive class. The model estimates the coefficients (weights) for each input 
# feature, and the predicted probability is calculated as a combination of 
# these weights and the input feature values.

# In logistic regression, the training process involves optimizing the model's c
# oefficients using an appropriate optimization algorithm, such as gradient descent.
# The objective is to find the coefficients that maximize the likelihood of the 
# observed data given the model. Once the model is trained, it can be used to
# predict the binary outcome for new data by calculating the probability and 
# applying a threshold (usually 0.5) to classify the data into the positive or negative class.

# Logistic regression has several advantages, including simplicity, interpretability, 
# and efficiency. It can handle both numerical and categorical input features, and it provides
# a probabilistic interpretation of the predictions. However, logistic regression assumes a 
# linear relationship between the input features and the log-odds of the outcome, which may 
# not hold in complex datasets. In such cases, more advanced algorithms like decision trees 
# or neural networks may be more suitable.

# ========================================================
# ========================================================
# ========================================================

# - Model Selection and Hyperparameter Tuning:
#   scikit-learn provides tools for model selection and hyperparameter 
#   tuning, such as cross-validation and grid search. Here's an example of
#   performing grid search for hyperparameter tuning of a support vector machine
#   (SVM) classifier:

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

X = np.array([[1, 0, -1], [-1, 1, 0], [0, -1, 1]])
y = np.array([1,0,1]) 

# Assuming X is your feature matrix and y is the target variable
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
model = SVC()

grid_search = GridSearchCV(model, param_grid, cv=3)
grid_search.fit(X, y)

best_params = grid_search.best_params_
