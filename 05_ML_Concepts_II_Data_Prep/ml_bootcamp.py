# %% [markdown]
# # Machine Learning Bootcamp

# %%
# Imports
import pandas as pd
import numpy as np 
#make sure to install sklearn in your terminal first!
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# %% [markdown]
# ## Phase I
# 
# ### Working to develop a model than can predict cereal quality rating...
# 
# -What is the target variable?
# 
# -Assuming we are able to optimize and make recommendations 
# how does this translate into a business context? 
# 
# -Prediction problem: Classification or Regression?
# 
# -Independent Business Metric: Assuming that higher ratings results in higher sales, can we predict which new cereals that enter the market over the next year will perform the best?

# %% [markdown]
# ## Phase II

# %% [markdown]
# ### Scale/Center/Normalizing/Variable Classes

# %%
#read in the cereal dataset, you should have this locally or you can use the URL linking to the class repo below
cereal = pd.read_csv("https://raw.githubusercontent.com/UVADS/DS-3001/main/data/cereal.csv")

cereal.info() # Let's check the structure of the dataset and see if we have any issues with variable classes
#usually it's converting things to category


# %%
#Looks like columns 11 and 12 need to be converted to category

cols = ["type","mfr","vitamins","shelf"]
cereal[cols]= cereal[cols].astype('category') 
#iloc accesses the index of a dataframe, bypassing having to manually type in the names of each column

#convert type variable in category variable
#cereal.type = cereal.type.astype('category') #this is the same as the above code, but for a single column


cereal.dtypes #another way of checking the structure of the dataset. Simpler, but does not give an index

# %%
#Let's take a closer look at mfr
print(cereal.mfr.value_counts()) #value_counts() simply displays variable counts as a vertical table.

# %%
#Usually don't want more than 5 groups, so we should collapse this factor  
#Keep the large groups (G, K) and group all the smaller categories as "Other"

top = ['K','G']
cereal.mfr = (cereal.mfr.apply(lambda x: x if x in top else "Other")).astype('category')
#lambda is a small anonymous function that can take any number of arguments but can only have one expression
#a simple lambda function is lambda a: a+10, if we passed 5 to this we would get back 15
#lambda functions are best used inside of another function, like in this example when it is used inside the apply function
#to use an if function in a lambda statement, the True return value comes first (x), then the if statement, then else, and then the False return

cereal.mfr.value_counts() #This is a lot better


# %%
cereal.type.value_counts() #looks good

# %%
cereal.vitamins.value_counts() #also good

# %%
cereal.weight.value_counts() #what about this one? ... Not a categorical variable groupings so it does not matter right now

# %% [markdown]
# ### Scale/Center

# %%
#Centering and Standardizing Data
sodium_sc = StandardScaler().fit_transform(cereal[['sodium']])
#reshapes series into an appropriate argument for the function fit_transform: an array

print(sodium_sc[:10]) #essentially taking the zscores of the data, here are the first 10 values

# %% [markdown]
# ### Normalizing the numeric values 

# %%
#Let's look at min-max scaling, placing the numbers between 0 and 1. 
sodium_n = MinMaxScaler().fit_transform(cereal[['sodium']])
print(sodium_n[:10])

# %%
#Let's check just to be sure the relationships are the same
cereal.sodium.plot.density()

# %%
pd.DataFrame(sodium_n).plot.density() #Checks out!

# %%
#Now we can move forward in normalizing the numeric values and create a index based on numeric columns:
abc = list(cereal.select_dtypes('number')) #select function to find the numeric variables and create a list  

cereal[abc] = MinMaxScaler().fit_transform(cereal[abc])
#print(cereal) #notice the difference in the range of values for the numeric variables

# %% [markdown]
# ### One-hot Encoding 

# %%
# Next let's one-hot encode those categorical variables

category_list = list(cereal.select_dtypes('category')) #select function to find the categorical variables and create a list  

cereal_1h = pd.get_dummies(cereal, columns = category_list) 
#get_dummies encodes categorical variables into binary by adding in indicator column for each group of a category and assigning it 0 if false or 1 if true
cereal_1h.info() #see the difference? This is one-hot encoding!

# %% [markdown]
# ### Baseline/Prevalance 

# %%
#This is essentially the target to which we are trying to out perform with our model. Percentage is represented by the positive class. 
#Rating is continuous, but we are going to turn it into a Boolean to be used for classification by selecting the top quartile of values.

print(cereal_1h.boxplot(column= 'rating', vert= False, grid=False))
print(cereal_1h.rating.describe()) #notice the upper quartile of values will be above 0.43 

# %%
#add this as a predictor instead of replacing the numeric version
cereal_1h['rating_f'] = pd.cut(cereal_1h.rating, bins = [-1,0.43,1], labels =[0,1])
#If we want two segments we input three numbers, start, cut and stop values

cereal_1h.info() #notice the new column rating_f, it is now category based on if the continuous value is above 0.43 or not

# %%
#So now let's check the prevalence 
prevalence = cereal_1h.rating_f.value_counts()[1]/len(cereal_1h.rating_f)
#value_count()[1] pulls the count of '1' values in the column (values above .43)

print(prevalence) #gives percent of values above .43 which is equivalent to the prevalence or our baseline

# %%
#let's just double check this
print(cereal_1h.rating_f.value_counts())
print(21/(21+56)) #looks good!

# %% [markdown]
# ### Dropping Variables and Partitioning   

# %%
#Divide up our data into three parts, Training, Tuning, and Test but first we need to...
#clean up our dataset a bit by dropping the original rating variable and the cereal name since we can't really use them

cereal_dt = cereal_1h.drop(['name','rating'],axis=1) #creating a new dataframe so we don't delete these columns from our working environment. 
print(cereal_dt)

# %%
# Now we partition
Train, Test = train_test_split(cereal_dt,  train_size = 55, stratify = cereal_dt.rating_f) 
#stratify perserves class proportions when splitting, reducing sampling error 

# %%
print(Train.shape)
print(Test.shape)

# %%
#Now we need to use the function again to create the tuning set
Tune, Test = train_test_split(Test,  train_size = .5, stratify= Test.rating_f)

# %%
#check the prevalance in all groups, they should be all equivalent in order to eventually build an accurate model
print(Train.rating_f.value_counts())
print(15/(40+15))

# %%
print(Tune.rating_f.value_counts())
print(3/(8+3)) #good

# %%
print(Test.rating_f.value_counts())
print(3/(8+3)) #same as above, good job!

# %% [markdown]
# # Now you try!

# %%
job = pd.read_csv("https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv")
print(job.head())

# %%
from io import StringIO
import requests

url="https://query.data.world/s/ttvvwduzk3hwuahxgxe54jgfyjaiul"
s=requests.get(url).text
c=pd.read_csv(StringIO(s))
print(c.head())
print(s)
# %%
job.info()
#summarize the missing values

# %%
#summarize the missing values in the job dataset
job.isna().sum()
#job.notna().sum() #this is the same as the above code, but for non-missing values

# %%
job1 = job.loc[job.notna().all(axis='columns')]
# %%
