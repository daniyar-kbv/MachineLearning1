# Info:

# Author:
# Date

# Purpose:

# inputs:

# outputs:

# Version control:

#------------------------------------------------------------------------------

import numpy as np

#          leave comments using "#"
#          or you can comment several lines with putting ''' at the begining and end of 
#    that section like:
'''
this line is comment
this line is comment
this line is comment
'''

#------------------------------------------------------------------------------
# String
# define a string
my_first_program = "Hello World"
# Note that you can use " or ' :   my_first_program = 'Hello World'

print (my_first_program)

# unexpected indent (just a warning)
    my_string = "Hello World!!!"

print (my_string)

# reassign the value of a string
my_string = "Hello!! World!!"
print (my_string)

# proper indent
a = my_string
if (a=="Hello World!!!"):
    print (a)

my_string = "Hellooooooooo!! World!!"
   
# how to comment/uncomment several lines in spyder?
# how to indent/unindent several lines in spyder?

# case sensitive
s=1
print (s, S)
# Print (s) or PRINT (s) will also return error 

#------------------------------------------------------------------------------
# Numbers, integer and flaot
# define an integer
a_integer = 1
print (a_integer)

# check the type
type(a_integer) # print (type(a_integer))


# define an float
a_float = 1.8
#a_float = 1.0 # or a_float = 1.0000
print (a_float)

# check the type
type(a_float)

# dont use python functions as names
#int = 1 # bad practice

# convert integers and float to each other
a_float = int (a_float)
a_integer = float(a_integer)

# add integer to a float
b = a_float + a_integer # convert all to float

#------------------------------------------------------------------------------
# convert string to numbers and vise versa
coffee = "3"
sandwich = "6"

total = sandwich + coffee
print (total) #print (sandwich + coffee)

total = int(sandwich) + (coffee) # mixed types

total = int(sandwich) + int(coffee)
print (total) #print (sandwich + coffee)

print ("total price is", total, "$")

new_string = str(3) # convert number to string
type (new_string)
print ("total price is " + str(total) + "$")

#------------------------------------------------------------------------------
# Boolean
z1 = 2
z2 = 3
B = (z1==z2)

print (B)

#------------------------------------------------------------------------------
# import libraries
x1 = 2
x_2 = sqrt (2) # error

import math
x_2 = math.sqrt (x1)

# OR
import math as mt
x_2 = mt.sqrt (x1)



#------------------------------------------------------------------------------
# Exercises:
'''
1: Create a float and an integer and add them together,
2: Create 2 strings, one as “my name is” and another with your name, then paste them and print them, 
3: Create string and numbers and add/paste them together (once as strings, once as numbers)
'''
s1= "Heloooo"
s2="worldddd"
new_s = s1+" "+s2
print(new_s)

# find the error?
c=1
b=2
d=C-b

#------------------------------------------------------------------------------
# other operations
x1=1.2
x2=3.5
print(x1 + x2)
print(x1 - x2)
print(x1 / x2)
print(x1 * x2)
print(x1 % x2)
print(x1 ** x2)
print(abs(x1))


#------------------------------------------------------------------------------
# Lists

my_first_list=[10, "Hi", 5]
print (my_first_list)
type(my_first_list)

my_first_list = [] # creat empty list
my_first_list.append(111) # append element 1
my_first_list.append(str(33))
my_first_list.append(2)
print (my_first_list)
type(my_first_list)

# indexing lists or slicing
print (my_first_list[0]) # indexing start at 0 in python

my_first_list[0] = 99

my_first_list=(10, "Hi", 5)


#------------------------------------------------------------------------------
# Conditionals



