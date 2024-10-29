#%%
print("Hello World")
#%%

#%%
#create a function that takes a list of numbers and returns the sum of the numbers
def sum_list(x):
    return sum(x)

import numpy as np
import pandas as pd

#%%
#function that computes the range of a variable and then, for no good reason, adds 100 and divides by 10.
def range_plus_100_div_10(x):
    return (max(x)-min(x)+100)/10

# %%
#create a list of numbers and then call the function on that list
x = [1,2,3,4,5,6,7,8,9,10]
range_plus_100_div_10(x)

# %%

