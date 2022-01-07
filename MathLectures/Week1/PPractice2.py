# Info:

# Author:
# Date

# Purpose:

# inputs:

# outputs:

# Version control:

#------------------------------------------------------------------------------

# leave comments using "#"
# or you can comment several lines with putting ''' at the begining and end of 
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
my_string  "Hello!! World!!"
print (my_string)

# proper indent
a = my_string
if (a=="Hello World!!!"):
    print (a)
    
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
a_float = 1.2
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
b = a_float+a_integer # convert all to float

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
# Python Prctices 2

# version of my python:

# Import library
import sys
print("My Python version is")
print (sys.version)
print("Version information:")
print (sys.version_info)

# print dat and time
import datetime
now = datetime.datetime.now()
print ("Current date and time is: ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))


#Write a code that calculate the area and perimeter of a circle, 
#       a rectangle and a right angle triangle
# circle
r = 2.0
pi = 3.14
area = pi*r**2
perim = 2*pi*r
print (r, perim)

# TRIANGLE
h = 2
b = 3
area = 1/2*b*h

# Volume of a sphere:
Vol = 4/3*pi*r**3

#Write a code that accept a name and a character and assign that number 
#      to that character
name = input('give me a name: ')
last_name = input ('give me the last name: ')

print (name, "last name is: ", last_name)

#Create a number like "b", then create an string like “bb” and “bbbb” 
#       then convert all back to numbers and add them together
b = 6
b_str = str(b)
x2=b_str+b_str
x3=b_str+b_str+b_str+b_str

final = b+x2+x3 # error, why?
final = b+int(x2)+int(x3)


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


