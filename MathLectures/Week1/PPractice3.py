# Info:

# Author:
# Date

# Purpose:

# inputs:

# outputs:

# Version control:


#------------------------------------------------------------------------------
# Lists

# Create an empty list
A_list = []

# Create a list of integers
A_list = [10, 20, 3333]


my_first_list=[10, "Hi", 5]
print (my_first_list)
type(my_first_list)

# indexing lists or slicing
print (my_first_list[0]) # indexing start at 0 in python

my_first_list[0] = 99
my_first_list=(10, "Hi", 5)


# Create a list with mixed datatypes, not homogeneous
B_list = [1, "Hello", 3.4]

my_first_list = [] # creat empty list
my_first_list.append(111) # append element 1
my_first_list.append(str(33))
my_first_list.append(2)
print (my_first_list)
type(my_first_list)


#List inside list:
B_list = [1, "Hello", 3.4, [2, 'hi'], A_list]

# Length of lists:
print (len(B_list))

#List slicing and indexing:
print(B_list[0])
print (B_list [1:3])
print (B_list [3])

# Negative Indexing
# Negative indexing means beginning from the end
B_list = [1, "Hello", 3.4, 5, 6]
print (B_list [-1])

#other indexing, slicing:
print (B_list [:3]) #python dont include the last index, in this case index 3
print (B_list [3:])
print (B_list [-3:-1])

#reassign values:
B_list [0] = "Hiiiii"
B_list [1] = 23

#Lists are a useful tool for creating a sequence and later iterating over it.
B_list = [1, "Hello", 3.4, 5, 6]

for i in B_list:
	print (i)

if 5 in B_list:
	print("list has the number 5!!!")

#Add or delete Items to/from lists
B_list.append(55)
B_list.extend([9, 90, 900])

My_list = ["Orange", "Banana", "Apple", "Mango"]
My_list.remove("Banana")

#Join 2 lists:
My_list = My_list+ My_list

#Copy a list:
Your_list = My_list.copy()

#------------------------------------------------------------------------------
# Tuples
my_tuple = ("Hello", "class", "2")
print (my_tuple)

# indexing
my_tuple[1]

my_tuple[0:3]
my_tuple[-1]

# assignment
my_tuple[1] = "1" # error? why?

t2 = my_tuple + my_tuple 
t2

len(t2)


#------------------------------------------------------------------------------
# Arrays
cities = ["Toronto", "London", "NYC", "Toronto"]

# create Arrays method 1: 
import array as arr
my_array_int = arr.array('i', [1,2,3,4,5])

my_array_float = arr.array('f', [1.1,2.1,3.1,4.1,5.1])

# create Arrays method 2: numpy library
import numpy as np
print(np.__version__)

my_array = np.array([1, 2, 3, 4, 5])

# Length of lists:
print (len(my_array))
print ((my_array.shape))

#List slicing and indexing:
print(my_array[0])
print (my_array [1:3])
print (my_array [3])

# Negative Indexing
print (my_array [-1])

#other indexing, slicing:
print (my_array [:3]) #python dont include the last index, in this case index 3
print (my_array [3:])
print (my_array [-3:-1])

#reassign values:
my_array [0] = "Hiiiii" # *** Error? why?
my_array [1] = 23

#Append arrays
my_array2 = np.append(my_array, my_array)

# Element wise array operations:

x = np.array([1, 2, 3, 4, 5])
y = np.array([2.3, 2.1, 30, 4.1, 6.6, 13])

type(x)
# what are the type of each? why?

z = x+y # ****Error? Why?

x = np.append(x,6)

z = x+y 
# What is type of z? why?
z = x*y 
z = x/y # ****Error? Why?

# arange function
a2 = np.arange(30,71,2)

# Create a 2D array:
a_2d = np.array([(1,2,3),(4,5,6)])

print (type (len(a_2d)) )
print (len(a_2d)) 

print (len(a_2d[0])) 
print (len(a_2d[1])) 

print (a_2d.shape[0])
print (a_2d.shape[1])


# Create an identity matrix
a_1 = np.identity(3)
print(a_1) 

a_2 = np.zeros((2,2))   # Create an array of all zeros
print(a_2)              

a_3 = np.ones((1,2))    # Create an array of all ones
print(a_3)              

a_4 = np.full((2,2), 7)  # Create a constant array
print(a_4)              

a_5 = np.eye(2)         # Create a 2x2 identity matrix
print(a_5) 

a_6 = np.random.random((4,4))  # Create an array filled with random values
print(a_6) 

a_6 = 10 * a_6
print(a_6) 

# 2-D Indexing
a_6 [0:3, 2:4]

# ## EXERCISE ## ================
    # create two 4*4 identity matrix
    # creat a 6*6 2darray (random) and a 2*2 array (eye), assign the 2by2 to 
    #       the middle of 6by6



#------------------------------------------------------------------------------
# Mean, Median and mode

a_mean = np.mean (y)
print ("the mean is:", a_mean)

a_median = np.median (y)
print ("the median is:", a_median)

# import lib
from scipy import stats

a_mode = stats.mode (y)
print ("the mode is:", a_mode)


# ## EXERCISE ## ================
    # find the mean, median and mode of whole array and different columns and rows

test_array = np.array([[1, 3, 4, 2, 2, 7],
                       [5, 2, 2, 1, 4, 1],
                       [3, 3, 2, 2, 1, 1]])

    
test_mean = np.mean(test_array, axis = 1)
print (test_mean)

test_mean = np.mean(test_array[0,:])

a_mode = stats.mode (test_array[:,1])
print (a_mode)


#------------------------------------------------------------------------------
# If conditions
x1 = 1
x2 = 2.5
if x1 > 10:
  print("value is greater than 10")
  
if x1 > 10:
    print("value is greater than 10")
else:
    print("value is not greater than 10")

if x1 > x2:
    print("value1 is greater than value2")
elif x1 == x2:
    print("two values are equal")
elif x2 < x1:
    print("value2 is greater than value1")





