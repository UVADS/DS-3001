# %% [markdown]
# # Trees Regression example

# %%
#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz 

from sklearn.model_selection import train_test_split,RepeatedKFold,GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor, export_graphviz 
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# %% [markdown]
# ## Load in same wine dataset from last week, let's preprocess!

# %%
#Read in the data from the github repo, you should also have this saved locally...
winequality = pd.read_csv("https://raw.githubusercontent.com/UVADS/DS-3001/main/data/winequality-red-ddl.csv")

# %%
#Let's take a look...
print(winequality.info()) #Some NA's

# %%
#drop NA's and create a new dataframe to preserve our working environment
winequality_1 = winequality.dropna()

# %%
#In order to use tree regression in python, all features must be numeric
print(winequality_1.dtypes)#Going to need to change text_rank

# %%
#Take a look at value counts beforehand 
print(winequality_1["text_rank"].value_counts())

# %%
#encode text_rank to become a continuous variable so it can be applied to sklearn's decision tree regressor function
winequality_1[["text_rank"]] = OrdinalEncoder().fit_transform(winequality_1[["text_rank"]])
print(winequality_1["text_rank"].value_counts()) #Looks good


# %%
print(winequality_1.info()) #check if all numeric, yep!

# %% [markdown]
# ## Splitting the Data

# %%
#split independent and dependent variables, alcohol will be our target variable in this example
X= winequality_1.drop(columns='alcohol')
y= winequality_1.alcohol

# %%
#Use train_test_split like always to get our train, test, and tune data sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=21)
X_tune, X_test, y_tune, y_test = train_test_split(X_test,y_test,  train_size = 0.50,random_state=49)

# %% [markdown]
# ## Let's Build the Model 

# %%
#Three steps in building a ML model
#Step 1: Cross validation process- The process by which the training data will be used to build the initial model must be set. 
#As seen below:

kf =RepeatedKFold(n_splits=10,n_repeats =5, random_state=42)
# number - number of folds
# repeats - number of times the CV is repeated, takes the average of these repeat rounds

# This essentially will split our training data into k groups. For each unique group it will hold out one as a test set
# and take the remaining groups as a training set. Then, it fits a model on the training set and evaluates it on the test set.
# Retains the evaluation score and discards the model, then summarizes the skill of the model using the sample of model evaluation scores we choose


# %%
#Step 2: Usually involves setting a hyper-parameter search. This is optional and the hyper-parameters vary by model. 
#Let's define our parameters, we can use any number, but let's start with only the max depth of the tree
#this will help us find a good balance between under fitting and over fitting in our model

param={
    "max_depth" : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    #"splitter":["best","random"],
    #"min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
    #"min_weight_fraction_leaf":[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    #"max_features":["auto","log2","sqrt",None],
    #"max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] 
    #'min_impurity_decrease':[0.00005,0.0001,0.0002,0.0005,0.001,0.0015,0.002,0.005,0.01],
    #'ccp_alpha':[.001,.01,.1]
        }

# %%
#What score do we want our model to be built on?
#Let's use rmse, r2, mae this time.
#print(metrics.SCORERS.keys()) #find them
#to get rmse looks like we will need to start with mse

# %%
#Define score, what score will the splits and parameters be judged by? Here we will pass several 
scoring= ['neg_mean_squared_error','r2','neg_mean_absolute_error']

# %%
#Step 3: Train the Model

#Regressor model we will use
reg=DecisionTreeRegressor(random_state=30)

#Set up search for best decisiontreeregressor estimator based on r-sqaured across all the different folds...
search = GridSearchCV(reg, param, scoring=scoring, n_jobs=-1, cv=kf,refit='r2')

#execute search on our trianing data
model = search.fit(X_train, y_train)

# %%
#Retrieve the best estimator out of all parameters passed, based on lowest room mean squared error ...
best= model.best_estimator_
print(best) #Depth of 7, good

# %%
#making the decision tree for the best estimator 
dot_data = export_graphviz(best, out_file =None,
               feature_names =X.columns, #feature names from dataset
               filled=True, 
                rounded=True, ) 
               
graph=graphviz.Source(dot_data)

# %%
#plotting it, if chunk does not produce visual, run specific line alone to print in console
graph #Pretty big, it is a continuous target afterall.

# %%
#What about the specific scores (rmse, r2, mae)? Let's try and extract them ...
print(model.cv_results_) #This is a dictionary and in order to extract info we need the keys

# %%
#Which one of these do we need?
print(model.cv_results_.keys()) #get mean_test and std_test for all of our scores, and will need our param_max_depth as well 

# %%
#Let's extract these scores based on depth!

#Scores: 

#Have negative mean squared error so need to take absolute value and square root to convert to root mean squared error
mean_sq_err = np.sqrt(abs(model.cv_results_['mean_test_neg_mean_squared_error']))
#nothing needs to be done for r2, it is the same
r2= model.cv_results_['mean_test_r2']
#Have negative mean absolute error so need to take absolute value
mae= abs(model.cv_results_['mean_test_neg_mean_absolute_error'])

#Get standard deviations as well...
SDmse =  model.cv_results_['std_test_neg_mean_squared_error']
SDr2= model.cv_results_['std_test_r2']
SDmae= model.cv_results_['std_test_neg_mean_absolute_error']

#Parameter:
depth= np.unique(model.cv_results_['param_max_depth']).data

#Build DataFrame:
final_model = pd.DataFrame(list(zip(depth, mean_sq_err, r2,mae, SDmse,SDr2,SDmae)),
               columns =['depth','rmse','r2','mae',"rmseSD",'r2SD','maeSD'])

print(final_model.head(10))
#Let's take a look in our variable explorer as well to get the full dataframe...

# %%
#Remember!
#If we used mutiple params... you won't be able to get the scores as easily
#Say we wanted to get the scores based on max_depth still, but this time we used the parameter ccp_alpha as well
#Use the np.where function to search for the indices where the other parameter equals their best result, in this case it is .001
#This is an example code to find mse: model.cv_results_['mean_test_neg_mean_squared_error'][np.where((model.cv_results_['param_ccp_alpha'] == .001))]

# %% [markdown]
# ## Let's see how we did.

# %%
#Depth of 7 does in fact have the best rmse (lowest)
print(plt.plot(final_model.depth, final_model.rmse))

# %%
#Best r2 is at 7 as well!
print(plt.plot(final_model.depth, final_model.r2))

# %%
print(best) #this matches our estimator, great!

# %% [markdown]
# ## Variable Importance

# %%
#Variable importance for the best estimator, how much did each variable affect the ultimate decision making?
varimp=pd.DataFrame(best.feature_importances_,index = X.columns,columns=['importance']).sort_values('importance', ascending=False)
print(varimp)

# %%
#Graph variable importance
plt.figure(figsize=(10,7))
print(varimp.importance.nlargest(7).plot(kind='barh')) #density had a huge impact!

# %% [markdown]
# ## Let's make some predictions now

# %%
#Remember, 'best' is our best estimator. Let's use it to make predictions on our test data
pred=best.predict(X_test)
print(pred[:10])

# %%
#This gives us accuracy, basically telling us how precise our model is
print(best.score(X_test, y_test))

# %%
print(best.score(X_tune, y_tune)) #Test data is covered a lot better ... could be a sign of over or under fitting

# %%
#This number is Rsquared which we want to be close to 1. 
print(metrics.r2_score(y_test, pred))

# %%
#We want this number, RSME, to be less than .5 
print(np.sqrt(metrics.mean_squared_error(y_test, pred))) 

# %%
#MAE we want to be below .5 as well
print(metrics.mean_absolute_error(y_test, pred)) #nice!

# %% [markdown]
# ## Pipeline Time!

# %%
#In this example we are going to use a pipeline for our preprocessing and test so our work can be further consolidated and easily replicated on any dataset
#To make things a little more interesting, let's impute values for NA's in our features instead of dropping them
#In order to do so we can't have any NA's in our target, let's check
print(winequality.alcohol.isna().sum())

# %%
#since there are NA's in our target, let's drop those rows and create a new dataframe to preserve our working environment
winequality_pipe=winequality.dropna(subset=['alcohol'])

# %%
#Split the features and the target
X1 = winequality_pipe.drop(columns="alcohol")
y1 = winequality_pipe.alcohol

# %%
#Get training, tuning, and test set
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, train_size=0.70, random_state=21)
X_tune1, X_test1, y_tune1, y_test1 = train_test_split(X_test1,y_test1,  train_size = 0.50,random_state=49)

# %%
#All prepocessing will be done inside of the pipeline, but the process will differ based on whether the feature is categorical or numerical
#So we must identify and divide these two types, let's see what we are working with
print(X1.dtypes)

# %%
#Now retrieve column labels for each type
numlist=X1.select_dtypes(np.float64).columns
catlist=X1.select_dtypes(object).columns

# %%
#Make pipeline for categorical data: first ordinal encode like before, then impute median of values for NA
catpipe= Pipeline([
    ('oh',OrdinalEncoder()),
    ('cat_imputer', SimpleImputer(strategy='median', missing_values= np.nan))
])

# %%
#Add together category pipeline with numerical preprocessing into a column transformer based on column type
#Let's impute median as well for the numerical NA's
#Since the numerical data only has one step for preprocessing we do not need to create a separate pipeline for it
data_pipeline = ColumnTransformer([
    ('numerical', SimpleImputer(strategy="median", missing_values= np.nan), numlist),
    ('categorical', catpipe, catlist),
])

# %%
#Combine the preprocessing with the model we wish to use, same model as before (DecisionTreeRegressor)
#make_pipeline simply names the steps automatically for us, with pipeline we must name them outselves as seen above.
pipe = make_pipeline(data_pipeline, reg)

# %%
#The params will be the same, but we have to reset them since in this pipeline example, the keys change in the resulting dictionary
params_pipe = {}
params_pipe["decisiontreeregressor__max_depth"] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

# %%
#Instead of just passing our model to the hyperparameter search, this time we pass the preprocessing as well
#This allows for preprocessing to occur on every fold which is computational heavier, but better simulates the real world
#In the previous example we didn't need this since we did not do any imputing, 
#but in this example we need it because the pipeline makes sure the imputed medians are unique and accurate to the specific folds
#Scoring and cross validation will be the same as before
search_pipe = GridSearchCV(pipe,params_pipe, cv= kf, scoring =scoring, refit= 'r2')

# %%
#Let's fit our model using our training data
model_pipe = search_pipe.fit(X_train1,y_train1) #may take a moment to run

# %%
#Let's get our results!
best_pipe= model_pipe.best_estimator_['decisiontreeregressor']
print(best_pipe) #Wow, a different depth!

# %%
#Let's see which one is better...
#First we have to preprocess our testing data using our column transformer to fit our model
X_test1p= data_pipeline.fit_transform(X_test1)
print(best_pipe.score(X_test1p,y_test1))

# %%
#Testing data got worse, but tuning data got better! 
X_tune1p= data_pipeline.fit_transform(X_tune1)
print(best_pipe.score(X_tune1p,y_tune1))

# %%
#RMSE got worse...
pred1=best_pipe.predict(X_test1p)
print(np.sqrt(metrics.mean_squared_error(y_test1, pred1)))

# %%
#Let's get the scores now
print(model_pipe.cv_results_.keys())

# %%
#Same as before, since only one parameter used, this will be rather straight forward, no need for np.where
mean_sq_err_p = np.sqrt(abs(model_pipe.cv_results_['mean_test_neg_mean_squared_error']))
r2_p= model_pipe.cv_results_['mean_test_r2']
mae_p= abs(model_pipe.cv_results_['mean_test_neg_mean_absolute_error'])

#Get standard deviations as well...
SDmse_p =  model_pipe.cv_results_['std_test_neg_mean_squared_error']
SDr2_p= model_pipe.cv_results_['std_test_r2']
SDmae_p= model_pipe.cv_results_['std_test_neg_mean_absolute_error']

depth_p= np.unique(model_pipe.cv_results_['param_decisiontreeregressor__max_depth'].data)


# %%
#take a look at the head and the whole dataframe in our variable explorer
final_model_p = pd.DataFrame(list(zip(depth_p, mean_sq_err_p, r2_p,mae_p, SDmse_p,SDr2_p,SDmae_p)),
               columns =['depth','rmse','r2','mae',"rmseSD",'r2SD','maeSD'])
print(final_model_p.head())
# %%
#Variable importance for the best estimator, how much did each variable affect the ultimate decision making?
varimp_p=pd.DataFrame(best_pipe.feature_importances_,index = X1.columns,columns=['importance']).sort_values('importance', ascending=False)
print(varimp_p) #text_rank not included this time.....
# %%
#Graph variable importance
plt.figure(figsize=(10,7))
print(varimp_p.importance.nlargest(7).plot(kind='barh')) #density has largest impact still
# %%
#time to make our tree visual
dot_data_p = export_graphviz(best_pipe, out_file =None,
               feature_names =X1.columns,
               filled=True, 
                rounded=True, ) 
               
graph_p=graphviz.Source(dot_data_p)
# %%
#take a look ... run the specific line below
graph_p # pretty similar, but let's stick with our original model since less error
